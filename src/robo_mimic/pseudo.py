"""Pseudo action labels: the action a video implies, and how wrong it is.

A teleoperated demo gives you (observation, action) pairs and costs a human
minute per robot minute. Human video is unlimited and has no action column at
all. A **pseudo action label** is that missing column, inferred from pixels.

WHAT WE EXTRACT, AND WHY EXACTLY THESE TWO SIGNALS
Dyna's DYNA-2 states its supervision plainly: "wrist poses for end-effector
trajectories, and a continuous grasp signal derived from the thumb-index
aperture" (dyna.co/dyna-2, 1M+ hours, zero robot data in pretraining). That is
the same pair `robo_mimic` already computes live -- `handframe.hand_pose` and
the pinch ratio. This module writes them down instead of executing them.

WHY THE ERROR MUST BE DECOMPOSED, NOT TOTALLED
"How accurate is a pseudo-label" has four independent answers, and lumping them
produces a number that cannot guide any decision:

    perception    pixels -> hand pose.            Fixable with better tracking.
    convention    hand pose -> desired tool pose. Fixable by deciding, once.
    embodiment    desired pose -> nearest the arm can reach. NOT fixable: the
                  arm has five joints and a pose has six numbers.
    execution     commanded joints -> actual joints. A control problem; we
                  measured 16.43 mm of gravity sag (phase-7-hardware.md).

Only the third is a property of the robot rather than of our software, and it
is the one people mean by "the embodiment gap". `decompose` reports them apart.

ON GROUND TRUTH
You cannot grade pseudo-labels against the teleop commands they produced --
those commands ARE the pseudo-labels, so the comparison is circular. Truth has
to arrive through a different channel than the camera. The SO-101 **leader
arm** is that channel: its encoders measure the operator's motion directly,
with no vision in the loop. Film the hand on the leader, extract labels from
the video, and compare against the encoders. `rigid_offset` solves for the
fixed hand-to-handle transform that separates the two frames.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .handframe import hand_pose
from .landmarks import HandLandmarks
from .project import plane_basis, project
from .types import F64, Pose

#: Tool frame origin to the tip of the moving jaw, metres. MEASURED from
#: model/scene.xml at the zero pose: 58.4 mm. This is the lever that turns an
#: orientation error into a position error where it actually matters -- at the
#: fingertips, not at the wrist. A 10 deg tool rotation is 10.2 mm of jaw.
JAW_LEVER_M = 0.0584

#: MediaPipe landmark indices for the aperture Dyna uses as its grasp signal.
THUMB_TIP = 4
INDEX_TIP = 8


@dataclass(frozen=True)
class PseudoAction:
    """One frame's inferred action. The two signals, and nothing else."""

    #: Tool orientation implied by the hand. Position is deliberately absent:
    #: a single camera cannot measure absolute position (ADR-002), so a
    #: trajectory is only ever relative to where it started.
    rotation: F64
    #: Thumb-index aperture divided by palm span, so hand size cancels.
    aperture: float
    timestamp_ms: int


@dataclass(frozen=True)
class ErrorBudget:
    """Where a pseudo-label's error comes from. Degrees, and mm at the jaw."""

    frames: int
    reachable: int
    #: Structural: how far the wanted roll axis points out of the arm's plane.
    #: No 5-DOF arm can honour this, at any position, ever.
    out_of_plane_deg: F64
    #: Joint limits, not structure: the pitch had to move to find a solution.
    pitch_shift_deg: F64
    #: Total orientation miss, after the projector has done its best.
    orientation_deg: F64

    @property
    def reachable_fraction(self) -> float:
        return self.reachable / self.frames if self.frames else 0.0

    @staticmethod
    def at_jaw_mm(degrees: float, lever_m: float = JAW_LEVER_M) -> float:
        """Chord length a rotation of `degrees` sweeps at the jaw tip."""
        return 2.0 * lever_m * float(np.sin(np.radians(degrees) / 2.0)) * 1000.0

    def summary(self) -> dict[str, float]:
        def p(v: F64, q: float) -> float:
            return float(np.percentile(v, q)) if len(v) else float("nan")

        return {
            "reachable_fraction": self.reachable_fraction,
            "out_of_plane_p50_deg": p(self.out_of_plane_deg, 50),
            "out_of_plane_p90_deg": p(self.out_of_plane_deg, 90),
            "out_of_plane_p90_mm": self.at_jaw_mm(p(self.out_of_plane_deg, 90)),
            "pitch_shift_p90_deg": p(self.pitch_shift_deg, 90),
            "orientation_p50_deg": p(self.orientation_deg, 50),
            "orientation_p90_deg": p(self.orientation_deg, 90),
            "orientation_p90_mm": self.at_jaw_mm(p(self.orientation_deg, 90)),
        }


def extract(hand: HandLandmarks) -> PseudoAction:
    """One frame of video -> the action it implies. Dyna's two signals."""
    pose = hand_pose(hand)
    span = float(np.linalg.norm(hand.world[9] - hand.world[0]))  # wrist->middle MCP
    gap = float(np.linalg.norm(hand.world[THUMB_TIP] - hand.world[INDEX_TIP]))
    aperture = gap / span if span > 1e-9 else 0.0
    return PseudoAction(pose.rotation, aperture, hand.timestamp_ms)


def structural_loss_deg(rotation: F64, pan_rad: float, axis: int = 0) -> float:
    """How far a wanted roll axis points out of the arm's plane, degrees.

    THE embodiment gap, isolated. The 5-DOF constraint is one equation --
    `a . n = 0`, the tool's roll axis must lie in the arm's plane (ADR-001) --
    so the angle between the wanted axis and that plane is exactly the part no
    amount of better perception, calibration or control can recover.

    Uses `abs`, which makes it invariant to whether the hand frame maps onto
    the tool frame parallel or antiparallel. That matters: those two conventions
    are 180 deg apart and ours has never been pinned down, but the structural
    loss is the same either way, so this number is safe to quote while the
    convention question is still open.
    """
    normal, _ = plane_basis(pan_rad)
    return float(np.degrees(np.arcsin(np.clip(abs(rotation[:, axis] @ normal), 0.0, 1.0))))


def decompose(
    actions: list[PseudoAction], position: F64, start: dict[str, float]
) -> ErrorBudget:
    """Run pseudo-labels through the projector and split the error up.

    `position` fixes the tool somewhere reachable so that POSITION never limits
    the result -- we are asking what the arm can do with the ORIENTATION it was
    handed, which is the part 5 DOF cannot fully honour.
    """
    out_of_plane, pitch_shift, orientation = [], [], []
    reachable = 0
    current = dict(start)
    for action in actions:
        projection = project(Pose(position, action.rotation), current=current)
        reachable += projection.reachable
        pan = np.radians(projection.joints["shoulder_pan"])
        out_of_plane.append(structural_loss_deg(action.rotation, float(pan)))
        pitch_shift.append(projection.pitch_shift_deg)
        orientation.append(projection.orientation_error_deg)
        current = projection.joints
    return ErrorBudget(
        frames=len(actions),
        reachable=reachable,
        out_of_plane_deg=np.asarray(out_of_plane, dtype=np.float64),
        pitch_shift_deg=np.asarray(pitch_shift, dtype=np.float64),
        orientation_deg=np.asarray(orientation, dtype=np.float64),
    )


def rigid_offset(source: F64, target: F64) -> tuple[F64, float]:
    """Best fixed rotation taking `source` frames onto `target` frames.

    For the leader-arm experiment. While the operator grips the handle their
    hand is rigidly attached to it, so the hand frame and the leader's tool
    frame differ by ONE constant rotation. Solving for it separates the fixed
    convention offset -- which is a decision we simply have not made -- from
    the per-frame perception error, which is the thing being measured.

    Kabsch over the stacked frames. Returns the offset and the residual angle
    in degrees after applying it: that residual is the perception error.
    """
    if len(source) != len(target) or len(source) == 0:
        raise ValueError("need equally many, and at least one, frame pair")
    # sum_i target_i @ source_i^T, whose polar factor is the best rotation
    correlation = np.einsum("nij,nkj->ik", target, source)
    u, _, vt = np.linalg.svd(correlation)
    flip = np.diag([1.0, 1.0, float(np.sign(np.linalg.det(u @ vt)))])
    offset = u @ flip @ vt

    residuals = []
    for s, t in zip(source, target, strict=True):
        relative = (offset @ s).T @ t
        cos = (float(np.trace(relative)) - 1.0) / 2.0
        skew = (relative - relative.T) / 2.0
        sin = float(np.linalg.norm(skew)) / np.sqrt(2.0)
        residuals.append(float(np.degrees(np.arctan2(sin, cos))))
    return offset, float(np.median(residuals))
