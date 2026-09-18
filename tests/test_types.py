"""Pose invariants, property-based.

`Pose` is the one place a malformed rotation can be caught. Everything after it
assumes orthonormality, so these tests are load-bearing rather than decorative.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from robo_mimic.types import Pose

# --- strategies ------------------------------------------------------------
finite = st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False)
angles = st.floats(min_value=-np.pi, max_value=np.pi, allow_nan=False, allow_infinity=False)
axes = st.lists(
    st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    min_size=3,
    max_size=3,
).filter(lambda v: np.linalg.norm(v) > 1e-3)


def _rodrigues(axis: list[float], angle: float) -> np.ndarray:
    """Build a guaranteed-valid rotation, so the strategy never wastes a draw."""
    unit = np.asarray(axis, float) / np.linalg.norm(axis)
    hat = np.array(
        [[0, -unit[2], unit[1]], [unit[2], 0, -unit[0]], [-unit[1], unit[0], 0]], float
    )
    return np.eye(3) + np.sin(angle) * hat + (1 - np.cos(angle)) * (hat @ hat)


poses = st.builds(
    lambda p, a, t: Pose(position=np.asarray(p, float), rotation=_rodrigues(a, t)),
    st.lists(finite, min_size=3, max_size=3),
    axes,
    angles,
)


# --- construction ----------------------------------------------------------
def test_identity_is_identity() -> None:
    pose = Pose.identity()
    assert np.array_equal(pose.position, np.zeros(3))
    assert np.array_equal(pose.rotation, np.eye(3))


@pytest.mark.parametrize(
    ("position", "rotation", "message"),
    [
        (np.zeros(2), np.eye(3), "position must be"),
        (np.zeros(3), np.eye(4), "rotation must be"),
        (np.array([np.nan, 0, 0]), np.eye(3), "non-finite"),
        (np.zeros(3), np.full((3, 3), 0.5), "not orthonormal"),
        (np.zeros(3), np.diag([-1.0, 1.0, 1.0]), "negative determinant"),
    ],
)
def test_bad_input_is_rejected(
    position: np.ndarray, rotation: np.ndarray, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        Pose(position=position, rotation=rotation)


# --- algebra ---------------------------------------------------------------
@given(poses)
@settings(max_examples=200, deadline=None)
def test_inverse_undoes_the_pose(pose: Pose) -> None:
    composed = pose @ pose.inverse()
    metres, degrees = composed.distance_to(Pose.identity())
    assert metres < 1e-12
    assert degrees < 1e-9


@given(poses, poses)
@settings(max_examples=200, deadline=None)
def test_composition_matches_the_4x4_product(a: Pose, b: Pose) -> None:
    """The class must not invent its own convention."""
    assert np.allclose((a @ b).matrix, a.matrix @ b.matrix, atol=1e-12)


@given(poses)
@settings(max_examples=200, deadline=None)
def test_matrix_round_trips(pose: Pose) -> None:
    back = Pose.from_matrix(pose.matrix)
    assert np.allclose(back.position, pose.position, atol=1e-15)
    assert np.allclose(back.rotation, pose.rotation, atol=1e-15)


@given(poses, poses)
@settings(max_examples=200, deadline=None)
def test_distance_is_symmetric_and_non_negative(a: Pose, b: Pose) -> None:
    forward = a.distance_to(b)
    backward = b.distance_to(a)
    assert forward[0] == pytest.approx(backward[0], abs=1e-12)
    assert forward[1] == pytest.approx(backward[1], abs=1e-12)
    assert forward[0] >= 0.0
    assert 0.0 <= forward[1] <= 180.0 + 1e-9


@given(poses)
@settings(max_examples=200, deadline=None)
def test_distance_to_self_is_zero(pose: Pose) -> None:
    metres, degrees = pose.distance_to(pose)
    assert metres == 0.0, "a pose is exactly where it is"
    assert degrees == 0.0, "and exactly as turned as it is -- see distance_to's docstring"


@given(poses)
@settings(max_examples=100, deadline=None)
def test_frozen(pose: Pose) -> None:
    with pytest.raises((AttributeError, TypeError)):
        pose.position = np.zeros(3)  # type: ignore[misc]


# --- regression: the defect hypothesis found on 2026-09-16 -----------------
def test_from_matrix_rejects_a_bad_shape() -> None:
    with pytest.raises(ValueError, match=r"expected \(4,4\)"):
        Pose.from_matrix(np.eye(3))


@pytest.mark.parametrize("true_deg", [1e-7, 1e-6, 1e-4, 1e-2, 1.0, 90.0, 179.0, 180.0])
def test_small_angles_are_accurate_not_merely_small(true_deg: float) -> None:
    """The regression test for `distance_to`'s numerics.

    The original `arccos((tr R - 1)/2)` form reported 5.73e-08 deg of error on a
    TRUE angle of 5.73e-08 deg -- 100 percent relative error -- and 1.71e-06 deg
    for a pose against itself. The `atan2` form is exact there. A relative bound
    is what pins that down; an absolute one would let the old code pass.
    """
    a = Pose.identity()
    b = Pose(position=np.zeros(3), rotation=_rodrigues([1.0, 2.0, 3.0], np.radians(true_deg)))
    _, measured = a.distance_to(b)
    assert measured == pytest.approx(true_deg, rel=1e-9, abs=1e-12)


def test_angle_never_exceeds_180() -> None:
    """atan2(|sin|, cos) is confined to [0, pi] by construction. Verify it."""
    generator = np.random.default_rng(1)
    for _ in range(2000):
        a = Pose(
            position=np.zeros(3),
            rotation=_rodrigues(list(generator.normal(size=3)), generator.uniform(-10, 10)),
        )
        b = Pose(
            position=np.zeros(3),
            rotation=_rodrigues(list(generator.normal(size=3)), generator.uniform(-10, 10)),
        )
        _, degrees = a.distance_to(b)
        assert 0.0 <= degrees <= 180.0
