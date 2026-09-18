"""Pose -> reachable joint angles, and an honest account of what was given up.

The Phase 3 gate lives here: the orientation the arm cannot deliver must be
confined to ONE direction, `c = n x a`, and the reported loss must match it.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from robo_mimic.kinematics import forward_kinematics, in_limits
from robo_mimic.kinematics.limits import JOINTS, LIMITS_DEG
from robo_mimic.project import (
    MAX_PITCH_SHIFT_DEG,
    Projection,
    pan_candidates,
    pitch_for,
    plane_basis,
    project,
    project_wrist_axis,
    wrist_axis,
)
from robo_mimic.types import Pose

from .conftest_hands import rodrigues

#: Feeding back a pose the arm is already in must return it untouched.
#: Measured over 400 random in-limit poses: 5.96e-16 m and 1.07e-12 deg.
ROUND_TRIP_M = 1e-12
ROUND_TRIP_DEG = 1e-9


def random_joints(rng: np.random.Generator) -> dict[str, float]:
    return {j: float(rng.uniform(*LIMITS_DEG[j])) for j in JOINTS}


def pose_of(joints: dict[str, float]) -> Pose:
    position, rotation = forward_kinematics({**joints, "gripper": 0.0})
    return Pose(position=position, rotation=rotation)


def residual_axis(rotation: np.ndarray) -> np.ndarray | None:
    skew = (rotation - rotation.T) / 2.0
    vector = np.array([skew[2, 1], skew[0, 2], skew[1, 0]])
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-10 else None


# --- the geometry, pinned to the FK -------------------------------------------
def test_plane_basis_matches_the_forward_kinematics(rng: np.random.Generator) -> None:
    """`n = (sin pan, cos pan, 0)`. Verified to 1.25e-16 when derived."""
    from robo_mimic.kinematics.forward import get_g12, get_gw1

    worst = 0.0
    for _ in range(200):
        pan_deg = float(rng.uniform(-110, 110))
        normal, radial = plane_basis(np.radians(pan_deg))
        frame = get_gw1(pan_deg) @ get_g12(0.0)  # type: ignore[no-untyped-call]
        truth = frame[:3, :3] @ np.array([0.0, 0.0, 1.0])
        worst = max(worst, float(np.abs(normal - truth).max()))
        assert abs(float(radial @ normal)) < 1e-12, "r must lie in the plane"
        assert abs(float(normal[2])) < 1e-15, "the plane normal is horizontal"
    assert worst < 1e-14


def test_the_wrist_axis_always_lies_in_the_arm_plane(rng: np.random.Generator) -> None:
    """The entire 5-DOF constraint, as one equation: `a . n = 0`.

    Measured over 3000 poses when derived: min = median = max = 0.000000000.
    """
    for _ in range(300):
        joints = random_joints(rng)
        pose = pose_of(joints)
        normal, _ = plane_basis(np.radians(joints["shoulder_pan"]))
        assert abs(float(wrist_axis(pose.rotation) @ normal)) < 1e-12


def test_pitch_formula_matches_tool_pitch(rng: np.random.Generator) -> None:
    """`pitch = pi - psi`. Verified to 1.02e-13 deg when derived."""
    from robo_mimic.kinematics import tool_pitch_of

    worst = 0.0
    for _ in range(300):
        joints = random_joints(rng)
        pose = pose_of(joints)
        normal, radial = plane_basis(np.radians(joints["shoulder_pan"]))
        axis, _ = project_wrist_axis(wrist_axis(pose.rotation), normal)
        predicted = pitch_for(axis, radial)
        truth = tool_pitch_of(
            joints["shoulder_lift"], joints["elbow_flex"], joints["wrist_flex"]
        )
        error = abs(float(np.mod(predicted - truth + np.pi, 2 * np.pi) - np.pi))
        worst = max(worst, error)
    assert np.degrees(worst) < 1e-9


# --- the projection itself ------------------------------------------------------
@given(
    axis=st.lists(st.floats(-1.0, 1.0), min_size=3, max_size=3).filter(
        lambda v: np.linalg.norm(v) > 0.2
    ),
    pan_deg=st.floats(-110.0, 110.0),
)
@settings(max_examples=300, deadline=None)
def test_projected_axis_is_in_plane_and_unit(axis: list[float], pan_deg: float) -> None:
    normal, _ = plane_basis(np.radians(pan_deg))
    unit = np.asarray(axis, float) / np.linalg.norm(axis)
    flattened, discarded_deg = project_wrist_axis(unit, normal)

    # The discarded angle is `arcsin(|a . n|)` in BOTH branches, degenerate or
    # not. An earlier version asserted exactly 90 deg in the degenerate branch,
    # which failed on axis = (0, 1, 1.19e-7): that is near the normal but not
    # on it, so the true answer is 89.999993 deg.
    assert discarded_deg == pytest.approx(
        abs(np.degrees(np.arcsin(np.clip(abs(unit @ normal), 0.0, 1.0)))), abs=1e-9
    )
    if not flattened.any():
        assert discarded_deg > 89.99, "only a near-normal axis may be degenerate"
        return
    assert float(np.linalg.norm(flattened)) == pytest.approx(1.0, abs=1e-12)
    # 1e-9, not 1e-12: normalising amplifies float error by 1/length, and
    # `DEGENERATE = 1e-6` bounds the shortest vector we are willing to normalise.
    assert abs(float(flattened @ normal)) < 1e-9


def test_a_wrist_axis_already_in_plane_is_untouched(rng: np.random.Generator) -> None:
    for _ in range(200):
        pan = float(rng.uniform(-np.pi, np.pi))
        normal, radial = plane_basis(pan)
        in_plane = np.cos(t := rng.uniform(-np.pi, np.pi)) * radial + np.sin(t) * np.array(
            [0.0, 0.0, 1.0]
        )
        flattened, discarded = project_wrist_axis(in_plane, normal)
        assert discarded == pytest.approx(0.0, abs=1e-9)
        assert np.abs(flattened - in_plane).max() < 1e-12


def test_an_axis_along_the_normal_is_degenerate_not_a_crash() -> None:
    """Pointing the wrist straight out of the plane: every in-plane direction is
    equally wrong, so there is no answer to pick. Report 90 deg, do not raise."""
    normal, _ = plane_basis(0.3)
    flattened, discarded = project_wrist_axis(normal, normal)
    assert not flattened.any()
    assert discarded == pytest.approx(90.0, abs=1e-6)


# --- round trip -----------------------------------------------------------------
def test_a_reachable_pose_comes_back_untouched(rng: np.random.Generator) -> None:
    """The strongest correctness check: the arm is already there, so nothing
    should be given up. Catches every sign error at once."""
    worst_m = worst_deg = 0.0
    for _ in range(200):
        joints = random_joints(rng)
        result = project(pose_of(joints), current=joints)
        assert result.status == "exact", result.status
        worst_m = max(worst_m, result.position_error_m)
        worst_deg = max(worst_deg, result.orientation_error_deg)
        assert result.out_of_plane_deg == pytest.approx(0.0, abs=1e-9)
        assert result.pitch_shift_deg == pytest.approx(0.0, abs=1e-9)
    assert worst_m < ROUND_TRIP_M, f"{worst_m:.3e} m"
    assert worst_deg < ROUND_TRIP_DEG, f"{worst_deg:.3e} deg"


def test_the_roll_sign_is_right(rng: np.random.Generator) -> None:
    """Regression. `wrist_roll` rotates about MINUS the tool x-axis: measured,
    +7 deg of joint gives exactly -7.0000 deg about `a`. Getting this backwards
    left the position exact and the orientation up to 180 deg wrong."""
    for _ in range(100):
        joints = random_joints(rng)
        result = project(pose_of(joints), current=joints)
        assert result.joints["wrist_roll"] == pytest.approx(
            joints["wrist_roll"], abs=1e-6
        )


# --- THE PHASE 3 GATE -----------------------------------------------------------
def test_the_discarded_rotation_is_confined_to_one_axis(
    rng: np.random.Generator,
) -> None:
    """THE GATE.

    Tilt a reachable pose about `c = n x a` -- the one direction the arm cannot
    rotate about -- and the projection must give back exactly that tilt and
    nothing else.

    NOT asserted as exact. The roll correction is a rotation about `a`, and
    rotations do not commute, so the total residual is a composition rather
    than a single rotation about `c`. Measured over ~4000 samples:

        |residual axis . c|        min 0.9937, median 0.9994
        |orientation error - tilt| p99 0.148 deg

    Bounds below are set from those measurements with room to spare.
    """
    alignments: list[float] = []
    mismatches: list[float] = []
    statuses: dict[str, int] = {}

    for _ in range(600):
        joints = random_joints(rng)
        pose = pose_of(joints)
        # The plane of the pose's OWN pan, not a guess from `pan_candidates`.
        # `pan_candidates` returns both branches and the pose may be on either;
        # using the wrong one leaves `a` up to 0.14 out of that plane, so the
        # cross product is not the forbidden axis and the tilt is not purely
        # out-of-plane. That mistake cost two wrong diagnoses.
        normal, _ = plane_basis(np.radians(joints["shoulder_pan"]))
        axis = wrist_axis(pose.rotation)
        assert abs(float(axis @ normal)) < 1e-12, "a must lie in ITS OWN plane"
        forbidden = np.cross(normal, axis)
        if float(np.linalg.norm(forbidden)) < 1e-6:
            continue
        forbidden /= np.linalg.norm(forbidden)

        tilt_deg = float(rng.uniform(-25.0, 25.0))
        target = Pose(
            position=pose.position,
            rotation=rodrigues(forbidden, np.radians(tilt_deg)) @ pose.rotation,
        )
        result = project(target, current=joints)
        statuses[result.status] = statuses.get(result.status, 0) + 1
        if result.status != "exact":
            continue

        assert result.position_error_m < 1e-9, "orientation loss must not leak into position"
        chosen_normal, _ = plane_basis(np.radians(result.joints["shoulder_pan"]))

        # The discarded angle IS the tilt -- but only when the branch project
        # chose shares the pose's plane. The two pan branches differ by
        # `2 * arccos(_LAT / h)`, which with `_LAT = -0.1778 mm` is ~0.1 deg off
        # antiparallel, so the same axis measured against the other branch's
        # normal gives a slightly different out-of-plane angle. Measured
        # discrepancy 0.049 deg. Real geometry, not noise.
        if abs(float(chosen_normal @ normal)) > 1.0 - 1e-9:
            assert result.out_of_plane_deg == pytest.approx(abs(tilt_deg), abs=1e-6)
        chosen = np.cross(chosen_normal, wrist_axis(result.achieved.rotation))
        if float(np.linalg.norm(chosen)) < 1e-6:
            continue
        chosen /= np.linalg.norm(chosen)

        residual = residual_axis(result.achieved.rotation @ target.rotation.T)
        if residual is None:
            continue
        alignments.append(abs(float(residual @ chosen)))
        mismatches.append(abs(result.orientation_error_deg - abs(tilt_deg)))

    assert len(alignments) > 400, f"too few exact samples: {statuses}"
    assert min(alignments) > 0.95, f"worst alignment {min(alignments):.6f}"
    assert float(np.median(alignments)) > 0.99
    assert float(np.percentile(mismatches, 99)) < 1.0


def test_out_of_plane_is_reported_even_when_large(rng: np.random.Generator) -> None:
    """A big tilt must be reported, not silently absorbed into position error."""
    for _ in range(60):
        joints = random_joints(rng)
        pose = pose_of(joints)
        pans = pan_candidates(pose.position)
        if not pans:
            continue
        normal, _ = plane_basis(pans[0])
        forbidden = np.cross(normal, wrist_axis(pose.rotation))
        if float(np.linalg.norm(forbidden)) < 1e-6:
            continue
        forbidden /= np.linalg.norm(forbidden)
        target = Pose(
            position=pose.position,
            rotation=rodrigues(forbidden, np.radians(60.0)) @ pose.rotation,
        )
        result = project(target, current=joints)
        if result.status == "unreachable":
            continue
        assert result.out_of_plane_deg > 30.0
        assert result.position_error_m < 1e-9, "orientation loss must not leak into position"


# --- totality: it must never fail -----------------------------------------------
@given(
    position=st.lists(st.floats(-0.6, 0.6), min_size=3, max_size=3),
    axis=st.lists(st.floats(-1.0, 1.0), min_size=3, max_size=3).filter(
        lambda v: np.linalg.norm(v) > 0.2
    ),
    angle=st.floats(-np.pi, np.pi),
)
@settings(max_examples=400, deadline=None)
def test_project_never_fails(
    position: list[float], axis: list[float], angle: float
) -> None:
    """Totality. Any pose in or out of the workspace yields a commandable,
    in-limits result. A solver that returns None mid-motion is a stutter."""
    target = Pose(position=np.asarray(position, float), rotation=rodrigues(axis, angle))
    result = project(target)
    assert isinstance(result, Projection)
    assert set(result.joints) == set(JOINTS)
    assert in_limits(result.joints), result.joints
    assert all(np.isfinite(v) for v in result.joints.values())
    assert np.isfinite(result.position_error_m)
    assert np.isfinite(result.orientation_error_deg)


def test_a_target_on_the_pan_axis_is_unreachable_not_a_crash() -> None:
    """The one place the arm genuinely cannot point: inside the tool's own
    0.178 mm lateral offset of its rotation axis."""
    from robo_mimic.kinematics.inverse import _D

    result = project(Pose(position=np.asarray(_D, float), rotation=np.eye(3)))
    assert result.status == "unreachable"
    assert in_limits(result.joints)
    assert np.isnan(result.out_of_plane_deg)
    assert not result.reachable


def test_a_far_away_target_is_unreachable_not_a_crash() -> None:
    result = project(Pose(position=np.array([2.0, 0.0, 0.0]), rotation=np.eye(3)))
    assert result.status == "unreachable"
    assert in_limits(result.joints)


def test_pan_candidates_are_empty_only_on_the_axis() -> None:
    from robo_mimic.kinematics.inverse import _D, _LAT

    assert pan_candidates(np.asarray(_D, float)) == []
    just_outside = np.asarray(_D, float) + np.array([2 * abs(_LAT), 0.0, 0.0])
    assert len(pan_candidates(just_outside)) == 2


# --- the honesty requirement ----------------------------------------------------
def test_status_never_claims_exact_when_something_was_given_up(
    rng: np.random.Generator,
) -> None:
    """Regression for a real defect.

    An earlier version, when the wanted roll fell outside `wrist_roll`'s range,
    silently reverted to roll = 0 and still reported "exact" -- a ~150 deg
    orientation error labelled as a perfect solve. 4 cases in 2982. The roll is
    now clamped and the status says `roll_clamped`.
    """
    for _ in range(400):
        joints = random_joints(rng)
        pose = pose_of(joints)
        target = Pose(
            position=pose.position,
            rotation=rodrigues(list(rng.normal(size=3)), rng.uniform(-np.pi, np.pi)),
        )
        result = project(target, current=joints)
        if result.status == "exact":
            assert result.pitch_shift_deg == pytest.approx(0.0, abs=1e-9)
            assert result.roll_clamp_deg == pytest.approx(0.0, abs=1e-9)


def test_pitch_shift_never_exceeds_its_bound(rng: np.random.Generator) -> None:
    for _ in range(200):
        joints = random_joints(rng)
        pose = pose_of(joints)
        target = Pose(
            position=pose.position,
            rotation=rodrigues(list(rng.normal(size=3)), rng.uniform(-np.pi, np.pi)),
        )
        result = project(target, current=joints)
        if result.reachable:
            assert result.pitch_shift_deg <= MAX_PITCH_SHIFT_DEG + 1e-9


def test_current_posture_breaks_ties_toward_staying_put(rng: np.random.Generator) -> None:
    """Branch continuity is a safety property: branches are far apart in joint
    space, so switching mid-motion slams the arm."""
    for _ in range(100):
        joints = random_joints(rng)
        pose = pose_of(joints)
        with_current = project(pose, current=joints)
        travel = max(abs(with_current.joints[j] - joints[j]) for j in JOINTS)
        assert travel < 1e-6, f"a reachable pose should not move the arm, moved {travel}"
