"""Desired pose -> joint angles the arm can actually reach. Pure, and total.

The module that was missing: `retarget` emits a `Pose`, the IK wants
`(position, pitch, roll)`, and nothing joined them up.

THE 5-DOF CONSTRAINT IS ONE EQUATION
    a . n = 0       a = the tool frame's x-axis,  n = the arm-plane normal

The tool's x-axis IS the wrist/roll axis, and it never leaves the arm's plane.
Everything here follows from that line.

So at a fixed position the achievable rotations are a 2-parameter family --
rotate about `n` (the pitch) and about `a` (the roll) -- and the ONE direction
you cannot have is `c = n x a`: in the plane, perpendicular to the wrist.

    This corrects ADR-001, which named the plane normal. Rotation about the
    normal IS the pitch, which is fully achievable.

THE GEOMETRY, all verified against the FK
    n = (sin pan, cos pan, 0)
    psi = atan2(a . z, a . r)        the wrist axis's angle within the plane
    pitch = pi - psi                 mod 2 pi

THREE LOSSES, REPORTED SEPARATELY RATHER THAN SUMMED
    out_of_plane_deg   structural. Five joints, six numbers. Unfixable.
    pitch_shift_deg    joint limits. Only ~14-20 percent of pitches are
                       reachable at a given position.
    roll_clamp_deg     `wrist_roll`'s ~40 deg dead sector.

Summing them would hide which one the operator can do something about.

NEVER FAILS
Always returns an in-limits, commandable result. A solver that returns None
mid-motion is a stutter.

Figures, and the four bugs the tests caught:
docs/measurements/phase-3-projection.md
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .kinematics import forward_kinematics, in_limits, solve_ik
from .kinematics.inverse import _D, _LAT, _RY180
from .kinematics.limits import JOINTS, clamp_deg
from .landmarks import F64
from .types import Pose

Status = Literal["exact", "pitch_shifted", "roll_clamped", "unreachable"]

#: Step for the outward pitch search, degrees.
PITCH_STEP_DEG = 1.0

#: How far the pitch may be shifted before a branch is abandoned, degrees.
#:
#: Measured in the real operating regime -- hold the anchor rotation, nudge the
#: position -- the gap between the wanted pitch and the nearest reachable one is
#: 0.0 deg at every scale from 2 mm to 100 mm of motion. The wanted pitch is
#: simply reachable. So this bound essentially never binds in normal use.
#:
#: It exists for the WRONG-BRANCH case. Each position has two pan branches, and
#: a given wrist axis is only achievable on one of them; on the other, no pitch
#: works. Unbounded, that branch swept the full period -- 361 IK calls, 23 ms
#: per frame. Bounded at 30 deg it costs 61 mostly-early-return calls.
#:
#: (For a RANDOM desired pitch the gaps are large: p50 49.5 deg, max 166 deg.
#: That is not the operating condition, and it is why orientation following is
#: off by default -- see ADR-003.)
MAX_PITCH_SHIFT_DEG = 30.0

#: A wrist axis this close to the plane normal has no meaningful in-plane
#: direction, so the projection is degenerate and we fall back to `r`.
#:
#: 1e-6, not 1e-9. Normalising a vector of length L amplifies float error by
#: 1/L, so a 1e-9 flattened vector came back only orthogonal to ~1e-7 -- and
#: hypothesis duly found a case (axis == normal to within 1e-12) where the
#: "in-plane" result was 1.1e-12 out of plane. At 1e-6 the residual is ~1e-10.
#: Physically this is an axis within arcsin(1e-6) = 6e-5 deg of the normal,
#: which discards ~90 deg either way, so nothing useful is lost by bailing.
DEGENERATE = 1e-6


@dataclass(frozen=True, slots=True)
class Projection:
    """What the arm will actually do, and what it had to give up.

    `joints` is always in-limits and always present -- a solver that returns
    nothing mid-motion is a stutter. When nothing is reachable, `status` is
    "unreachable" and `joints` holds the nearest attempt, with the errors saying
    how far off it is.
    """

    joints: dict[str, float]
    achieved: Pose
    status: Status
    #: Metres between the requested and achieved tool position.
    position_error_m: float
    #: Degrees between the requested and achieved tool orientation, total.
    orientation_error_deg: float
    #: The STRUCTURAL part: how far the wanted wrist axis pointed out of the
    #: arm plane. No 5-DOF arm can honour this.
    out_of_plane_deg: float
    #: The JOINT-LIMIT part: how far the pitch had to move to find a solution.
    pitch_shift_deg: float
    #: How far the wanted roll had to be clamped to stay inside `wrist_roll`'s
    #: range. Reported rather than swallowed: an earlier version silently
    #: reverted to roll = 0 here, producing a ~150 deg orientation error while
    #: still claiming status "exact". That was 4 cases in 2982.
    roll_clamp_deg: float = 0.0

    @property
    def reachable(self) -> bool:
        return self.status != "unreachable"


def pan_candidates(position: F64) -> list[float]:
    """The two pan angles that put the tool at `position`, radians.

    Replicates the IK's own pan solve, because the arm plane -- and therefore
    the pitch -- must be known BEFORE the IK is called. Empty when the target is
    inside the tool's lateral offset of the pan axis, which is the one place the
    arm genuinely cannot point.
    """
    u = _RY180 @ (np.asarray(position, float) - _D)
    a, b, c = u[1], -u[0], _LAT
    h = float(np.hypot(a, b))
    if h < abs(c):
        return []
    base = float(np.arctan2(b, a))
    offset = float(np.arccos(np.clip(c / h, -1.0, 1.0)))
    return [base + offset, base - offset]


def plane_basis(pan_rad: float) -> tuple[F64, F64]:
    """(plane normal, in-plane horizontal) for a pan angle.

    Verified against the FK to 1.25e-16. The plane also contains world z, so
    `(r, z)` is an orthonormal basis for it.
    """
    n = np.array([np.sin(pan_rad), np.cos(pan_rad), 0.0])
    r = np.array([np.cos(pan_rad), -np.sin(pan_rad), 0.0])
    return n, r


def wrist_axis(rotation: F64) -> F64:
    """The tool frame's x-axis: the wrist/roll axis, and the constrained one."""
    return np.asarray(rotation[:, 0], dtype=np.float64)


def project_wrist_axis(axis: F64, normal: F64) -> tuple[F64, float]:
    """Flatten a wanted wrist axis into the arm plane.

    Returns the in-plane unit axis and the angle that had to be discarded. That
    angle is the structural 5-DOF loss, and the correction is a pure rotation
    about `normal x axis` -- the missing direction identified above.
    """
    out_of_plane = float(axis @ normal)
    flattened = axis - out_of_plane * normal
    length = float(np.linalg.norm(flattened))
    if length < DEGENERATE:
        # The wanted axis points straight out of the plane; no in-plane
        # direction is closer than any other. 90 degrees is discarded either way.
        return np.zeros(3), float(np.degrees(np.arcsin(min(1.0, abs(out_of_plane)))))
    return flattened / length, float(np.degrees(np.arcsin(np.clip(abs(out_of_plane), 0.0, 1.0))))


def pitch_for(axis_in_plane: F64, radial: F64) -> float:
    """In-plane wrist axis -> the IK's `pitch`, radians.

    `pitch = pi - psi` where `psi = atan2(a.z, a.r)`. Verified against
    `tool_pitch` to 1.02e-13 degrees.
    """
    psi = float(np.arctan2(axis_in_plane[2], axis_in_plane @ radial))
    return float(np.pi - psi)


def _axis_angle(rotation: F64, axis: F64) -> float:
    """Signed angle of `rotation` measured about `axis`, radians.

    `atan2` form, not `arccos`: see `Pose.distance_to` for why the latter loses
    half its precision near zero, which is exactly where roll usually sits.
    """
    skew = (rotation - rotation.T) / 2.0
    vector = np.array([skew[2, 1], skew[0, 2], skew[1, 0]])
    return float(np.arctan2(vector @ axis, (float(np.trace(rotation)) - 1.0) / 2.0))


def _pitch_sweep(desired: float, step_deg: float, bound_deg: float) -> list[float]:
    """The wanted pitch first, then outward in both directions, up to `bound_deg`.

    Outward order is what makes "nearest reachable" true by construction rather
    than by search: the first in-limits hit IS the closest one.
    """
    step = np.radians(step_deg)
    count = max(1, int(round(bound_deg / step_deg)))
    out = [desired]
    for k in range(1, count + 1):
        out.append(desired + k * step)
        out.append(desired - k * step)
    return out


def project(
    target: Pose,
    current: dict[str, float] | None = None,
    *,
    step_deg: float = PITCH_STEP_DEG,
    max_pitch_shift_deg: float = MAX_PITCH_SHIFT_DEG,
) -> Projection:
    """Nearest pose the arm can actually reach, plus what it cost.

    Never raises and never returns None. `current` biases branch choice toward
    the arm's present posture, because different IK branches are far apart in
    joint space and switching between them mid-motion slams the arm.
    """
    best: Projection | None = None
    for pan in _ordered_pans(target.position, current):
        normal, radial = plane_basis(pan)
        axis, out_of_plane_deg = project_wrist_axis(wrist_axis(target.rotation), normal)
        if not axis.any():
            axis = radial
        wanted_pitch = pitch_for(axis, radial)

        for pitch in _pitch_sweep(wanted_pitch, step_deg, max_pitch_shift_deg):
            legal = [s for s in solve_ik(target.position, pitch, 0.0) if in_limits(s)]
            if not legal:
                continue
            # Keep only solutions on THIS pan branch: the other branch has its
            # own plane, hence its own pitch, and is handled by the outer loop.
            legal = [s for s in legal
                     if abs(_wrap_deg(s["shoulder_pan"] - np.degrees(pan))) < 1e-6]
            if not legal:
                continue

            shift_deg = abs(np.degrees(pitch - wanted_pitch))
            for solution in legal:
                candidate = _finish(solution, target, axis, out_of_plane_deg, shift_deg)
                if best is None or _better(candidate, best, current):
                    best = candidate
            break  # first in-limits pitch is the nearest by construction

        if best is not None and best.status == "exact":
            # Nothing on the other branch can beat exact, and `_ordered_pans`
            # already put the least-travel branch first.
            break

    if best is not None:
        return best
    return _unreachable(target)


def _wrap_deg(degrees: float) -> float:
    return (degrees + 180.0) % 360.0 - 180.0


def _ordered_pans(position: F64, current: dict[str, float] | None) -> list[float]:
    """Pan branches, nearest the arm's present pan first.

    A given wrist axis is achievable on only one branch, so trying the wrong one
    first means sweeping its whole pitch bound for nothing. Measured: the far
    branch won 0 of 120 live frames while costing 61 of the 62 IK calls.
    """
    pans = pan_candidates(position)
    if current is None or len(pans) < 2:
        return pans
    here = current["shoulder_pan"]
    return sorted(pans, key=lambda pan: abs(_wrap_deg(np.degrees(pan) - here)))


def _finish(
    solution: dict[str, float],
    target: Pose,
    axis: F64,
    out_of_plane_deg: float,
    pitch_shift_deg: float,
) -> Projection:
    """Pick the roll that best matches the target, then measure what we got.

    With the pitch fixed, the achieved and wanted rotations share a wrist axis,
    so they differ by a single rotation about it. That angle IS the roll, read
    off directly rather than searched for.
    """
    zero_roll = dict(solution, wrist_roll=0.0)
    position, rotation = forward_kinematics({**zero_roll, "gripper": 0.0})
    # NEGATED. Measured: +7 deg of `wrist_roll` rotates the tool by exactly
    # -7.0000 deg about `a`. The tool's x-axis is MINUS frame 4's y, which is
    # the joint's own axis, so the joint and the measured angle run opposite.
    roll_deg = -float(np.degrees(_axis_angle(target.rotation @ rotation.T, axis)))

    wanted = _wrap_deg(roll_deg)
    # CLAMP, do not revert. `wrist_roll` spans -157.2..162.8 deg, so a wrapped
    # roll can fall in the ~40 deg dead sector behind the joint. Clamping keeps
    # the nearest legal orientation; reverting to 0 threw the whole roll away.
    used = clamp_deg("wrist_roll", wanted)
    final = dict(solution, wrist_roll=used)
    position, rotation = forward_kinematics({**final, "gripper": 0.0})

    achieved = Pose(position=position, rotation=rotation)
    metres, degrees = achieved.distance_to(target)
    roll_clamp_deg = abs(wanted - used)
    if pitch_shift_deg >= 1e-9:
        status: Status = "pitch_shifted"
    elif roll_clamp_deg >= 1e-9:
        status = "roll_clamped"
    else:
        status = "exact"
    return Projection(
        joints={j: float(final[j]) for j in JOINTS},
        achieved=achieved,
        status=status,
        position_error_m=metres,
        orientation_error_deg=degrees,
        out_of_plane_deg=out_of_plane_deg,
        pitch_shift_deg=pitch_shift_deg,
        roll_clamp_deg=roll_clamp_deg,
    )


def _better(
    candidate: Projection, incumbent: Projection, current: dict[str, float] | None
) -> bool:
    """Prefer less given up; break ties by staying near the current posture."""
    if candidate.pitch_shift_deg != incumbent.pitch_shift_deg:
        return candidate.pitch_shift_deg < incumbent.pitch_shift_deg
    if abs(candidate.roll_clamp_deg - incumbent.roll_clamp_deg) > 1e-9:
        return candidate.roll_clamp_deg < incumbent.roll_clamp_deg
    if abs(candidate.orientation_error_deg - incumbent.orientation_error_deg) > 1e-9:
        return candidate.orientation_error_deg < incumbent.orientation_error_deg
    if current is None:
        return False
    return _travel(candidate, current) < _travel(incumbent, current)


def _travel(projection: Projection, current: dict[str, float]) -> float:
    """Largest single-joint move from `current` to this projection, degrees."""
    return max(abs(projection.joints[j] - current[j]) for j in JOINTS)


def _unreachable(target: Pose) -> Projection:
    """Nothing in limits anywhere. Report it rather than returning None.

    The joints are left at zero: a defined, safe posture. The caller decides
    what to do, and the safety layer will refuse to jump there.
    """
    joints = dict.fromkeys(JOINTS, 0.0)
    position, rotation = forward_kinematics({**joints, "gripper": 0.0})
    achieved = Pose(position=position, rotation=rotation)
    metres, degrees = achieved.distance_to(target)
    return Projection(
        joints=joints,
        achieved=achieved,
        status="unreachable",
        position_error_m=metres,
        orientation_error_deg=degrees,
        out_of_plane_deg=float("nan"),
        pitch_shift_deg=float("nan"),
        roll_clamp_deg=float("nan"),
    )


__all__ = [
    "MAX_PITCH_SHIFT_DEG",
    "PITCH_STEP_DEG",
    "Projection",
    "pan_candidates",
    "pitch_for",
    "plane_basis",
    "project",
    "project_wrist_axis",
    "wrist_axis",
]
