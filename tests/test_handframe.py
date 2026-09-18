"""The hand frame: orthonormal always, and not mirrored between hands."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from robo_mimic.handframe import (
    DegenerateHandError,
    hand_pose,
    image_position,
    image_span,
)
from robo_mimic.landmarks import (
    INDEX_MCP,
    MIDDLE_MCP,
    PALM,
    PINKY_MCP,
    WRIST,
    HandLandmarks,
)

from .conftest_hands import make_hand, rodrigues

MIRROR_X = np.diag([-1.0, 1.0, 1.0])

angles = st.floats(min_value=-np.pi, max_value=np.pi, allow_nan=False, allow_infinity=False)
axes = st.lists(
    st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    min_size=3,
    max_size=3,
).filter(lambda v: np.linalg.norm(v) > 1e-3)


# --- the answer is known exactly --------------------------------------------
def test_canonical_hand_gives_the_identity_frame() -> None:
    """A flat right hand, fingers +y, palm facing +z, thumb side -x.

    Built so the frame must come out as exactly I. If any sign or cross-product
    order is wrong this fails immediately, and says which axis.
    """
    rotation = hand_pose(make_hand()).rotation
    assert np.abs(rotation - np.eye(3)).max() == 0.0


def test_frame_rotates_with_the_hand() -> None:
    """Turn the hand by R, the frame turns by R. Nothing else should change."""
    for angle in (0.3, 1.0, 2.5):
        turn = rodrigues((0.3, -0.7, 0.6), angle)
        rotated = hand_pose(make_hand(rotation=turn)).rotation
        assert np.abs(rotated - turn).max() < 1e-12


# --- the Phase 2 gate: orthonormal under fuzz -------------------------------
@given(axis=axes, angle=angles, pinch=st.floats(0.05, 1.5), side=st.sampled_from(["Left", "Right"]))
@settings(max_examples=400, deadline=None)
def test_frame_is_always_orthonormal(
    axis: list[float], angle: float, pinch: float, side: str
) -> None:
    """Any pose, any hand, any finger position -> a proper rotation.

    `Pose.__post_init__` also checks this, so a failure here means the frame
    construction produced something that is not a rotation at all.
    """
    rotation = hand_pose(
        make_hand(rotation=rodrigues(axis, angle), pinch_ratio=pinch, handedness=side)
    ).rotation
    assert np.abs(rotation.T @ rotation - np.eye(3)).max() < 1e-12
    assert float(np.linalg.det(rotation)) == pytest.approx(1.0, abs=1e-12)


@given(axis=axes, angle=angles)
@settings(max_examples=200, deadline=None)
def test_orthonormal_even_on_the_real_fixture_hands(
    golden_hands: list[HandLandmarks], axis: list[float], angle: float
) -> None:
    """Same property, but on photographed hands, whose palm triangle is skewed.

    Measured: `forward . across` is -0.26 and -0.42 on these, i.e. 75 and 115
    degrees rather than 90. Orthonormality must survive that.
    """
    turn = rodrigues(axis, angle)
    for hand in golden_hands:
        rotated = HandLandmarks(
            image=hand.image,
            world=hand.world @ turn.T,
            handedness=hand.handedness,
            score=hand.score,
            timestamp_ms=0,
        )
        rotation = hand_pose(rotated).rotation
        assert np.abs(rotation.T @ rotation - np.eye(3)).max() < 1e-12


def test_the_palm_triangle_really_is_skewed(golden_hands: list[HandLandmarks]) -> None:
    """Documents why the cross product is load-bearing rather than decorative."""
    for hand in golden_hands:
        forward = hand.world[MIDDLE_MCP] - hand.world[WRIST]
        across = hand.world[PINKY_MCP] - hand.world[INDEX_MCP]
        cosine = float(
            forward @ across / (np.linalg.norm(forward) * np.linalg.norm(across))
        )
        assert abs(cosine) > 0.15, "if these were perpendicular the fix-up would be moot"


# --- the mirror test: handedness ---------------------------------------------
@given(axis=axes, angle=angles)
@settings(max_examples=200, deadline=None)
def test_mirroring_a_hand_mirrors_its_frame(axis: list[float], angle: float) -> None:
    """Reflect a right hand and you get a physically left hand.

    Under a reflection M, real direction vectors map through M. The fingers
    direction (y) and the palm normal (z) are real directions, so both do. The
    third axis cannot: a mirror reverses handedness, so x picks up an extra
    sign. That asymmetry is the test -- it fails if the sign correction is
    missing, wrong, or applied to both hands.
    """
    turn = rodrigues(axis, angle)
    right = make_hand(rotation=turn, handedness="Right")
    left = HandLandmarks(
        image=right.image,
        world=right.world @ MIRROR_X.T,
        handedness="Left",
        score=right.score,
        timestamp_ms=0,
    )
    right_rotation = hand_pose(right).rotation
    left_rotation = hand_pose(left).rotation

    for column, sign in ((0, -1.0), (1, +1.0), (2, +1.0)):
        expected = sign * (MIRROR_X @ right_rotation[:, column])
        assert np.abs(left_rotation[:, column] - expected).max() < 1e-12


def test_without_the_sign_correction_the_frames_would_be_mirrored(
    golden_hands: list[HandLandmarks],
) -> None:
    """The bug this guards against, stated as an executable fact."""
    from robo_mimic.handframe import _PALM_NORMAL_SIGN, _orthonormal

    assert _PALM_NORMAL_SIGN["Left"] == -_PALM_NORMAL_SIGN["Right"]
    hand = golden_hands[0]
    forward = hand.world[MIDDLE_MCP] - hand.world[WRIST]
    across = hand.world[PINKY_MCP] - hand.world[INDEX_MCP]
    plus = _orthonormal(forward, across, +1.0)
    minus = _orthonormal(forward, across, -1.0)
    assert np.abs(plus[:, 2] + minus[:, 2]).max() < 1e-12, "the two signs must oppose"


# --- position and scale -------------------------------------------------------
def test_palm_centre_is_the_knuckle_mean(golden_hands: list[HandLandmarks]) -> None:
    for hand in golden_hands:
        assert np.allclose(hand_pose(hand).position, hand.world[list(PALM)].mean(axis=0))


def test_image_position_is_linear_in_screen_offset() -> None:
    """Move the hand across the frame, the proxy moves proportionally."""
    base = image_position(make_hand(image_centre=(0.5, 0.5), image_span=0.2))
    right = image_position(make_hand(image_centre=(0.7, 0.5), image_span=0.2))
    assert right[0] - base[0] == pytest.approx(0.2 / 0.2, abs=1e-12)
    assert right[1] == pytest.approx(base[1], abs=1e-12)
    assert right[2] == pytest.approx(base[2], abs=1e-12)


def test_depth_falls_as_the_hand_approaches() -> None:
    """Bigger on screen means nearer, so the depth coordinate must shrink."""
    near = image_position(make_hand(image_span=0.4))
    far = image_position(make_hand(image_span=0.2))
    assert near[2] < far[2]
    assert far[2] / near[2] == pytest.approx(2.0, rel=1e-12)


@given(span=st.floats(0.05, 0.45))
@settings(max_examples=200, deadline=None)
def test_moving_one_apparent_palm_span_always_reads_as_exactly_one(span: float) -> None:
    """The invariance ADR-002 promises: no per-operator calibration.

    A big hand far away and a small hand close up fill the same pixels. Both,
    moved by one of their OWN hand-lengths, shift on screen by one apparent
    span. Dividing by that span turns both into 1.0 -- so the same
    `scale_m_per_span` suits every operator, whatever size their hands are.

    (The earlier version of this test took a `scale` parameter and never used
    it, comparing a value to itself. Ruff's unused-argument rule caught it.)
    """
    base = image_position(make_hand(image_centre=(0.5, 0.5), image_span=span))
    moved = image_position(make_hand(image_centre=(0.5 + span, 0.5), image_span=span))
    assert moved[0] - base[0] == pytest.approx(1.0, rel=1e-12)
    assert moved[2] == pytest.approx(base[2], rel=1e-12), "sideways motion is not depth"


def test_aspect_correction_stretches_only_the_vertical() -> None:
    hand = make_hand(image_centre=(0.5, 0.5))
    square = image_position(hand, aspect=1.0)
    wide = image_position(hand, aspect=720 / 1280)
    assert wide[0] == pytest.approx(square[0], abs=1e-12)
    assert wide[2] == pytest.approx(square[2], abs=1e-12)
    assert wide[1] == pytest.approx(square[1] * 720 / 1280, abs=1e-12)


# --- degenerate input ---------------------------------------------------------
def test_collapsed_palm_raises() -> None:
    hand = make_hand()
    broken = HandLandmarks(
        image=hand.image,
        world=np.zeros_like(hand.world),
        handedness="Right",
        score=0.9,
        timestamp_ms=0,
    )
    with pytest.raises(DegenerateHandError, match="wrist and middle knuckle coincide"):
        hand_pose(broken)


def test_collinear_palm_raises() -> None:
    """Knuckles on the same line as the fingers: the normal is undefined."""
    hand = make_hand()
    world = hand.world.copy()
    world[INDEX_MCP] = world[MIDDLE_MCP] + np.array([0.0, 0.01, 0.0])
    world[PINKY_MCP] = world[MIDDLE_MCP] - np.array([0.0, 0.01, 0.0])
    broken = HandLandmarks(
        image=hand.image, world=world, handedness="Right", score=0.9, timestamp_ms=0
    )
    with pytest.raises(DegenerateHandError, match="collinear"):
        hand_pose(broken)


def test_zero_size_hand_on_screen_raises() -> None:
    hand = make_hand()
    image = hand.image.copy()
    image[WRIST] = image[MIDDLE_MCP]
    broken = HandLandmarks(
        image=image, world=hand.world, handedness="Right", score=0.9, timestamp_ms=0
    )
    assert image_span(broken) == 0.0
    with pytest.raises(DegenerateHandError, match="zero apparent size"):
        image_position(broken)


# --- regression: found by live testing, 2026-09-17 ------------------------------
def test_pure_depth_motion_does_not_invent_lateral_motion() -> None:
    """THE BUG: the proxy divided `cx` by the span instead of `cx - 0.5`.

    Only differences are used, so a constant offset looks harmless -- but it does
    NOT cancel when the span changes, because the span divides the offset.
    Measured on a hand held dead centre and moved only in depth:

        span x1.0   lateral drift 0.000 spans
        span x1.2                 0.589
        span x1.5                 1.179
        span x2.0                 1.768      <- roughly 18 cm of tool motion

    Pure depth motion was generating lateral motion. Offsets are now measured
    from the optical axis, where `(0.5 - 0.5)/span` is zero for every span.
    """
    reference = None
    for scale in (1.0, 1.2, 1.5, 2.0, 3.0):
        hand = make_hand(image_centre=(0.5, 0.5), image_span=0.2 * scale)
        proxy = image_position(hand, aspect=1.0)
        if reference is None:
            reference = proxy
            continue
        drift = float(np.hypot(*(proxy[:2] - reference[:2])))
        assert drift == 0.0, f"span x{scale}: {drift:.3e} spans of phantom lateral motion"


@given(
    span=st.floats(0.05, 0.45),
    offset=st.floats(-0.3, 0.3),
)
@settings(max_examples=200, deadline=None)
def test_a_hand_on_the_optical_axis_has_zero_lateral_proxy(
    span: float, offset: float
) -> None:
    """The invariant behind the fix, stated directly: on the axis, zero."""
    centred = image_position(make_hand(image_centre=(0.5, 0.5), image_span=span))
    assert abs(float(centred[0])) < 1e-12
    assert abs(float(centred[1])) < 1e-12

    # And off-axis motion still scales with 1/span, as the geometry requires.
    moved = image_position(make_hand(image_centre=(0.5 + offset, 0.5), image_span=span))
    assert float(moved[0]) == pytest.approx(offset / span, rel=1e-12)
