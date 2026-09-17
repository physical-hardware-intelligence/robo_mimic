"""HandLandmarks / Frame: pure, and driven entirely from the committed fixture.

Not one test here imports mediapipe. That is the point of Phase 1: once the
landmarks are frozen to a file, every downstream stage is testable in CI on a
machine with no camera, no GPU, and no 60 MB native dependency.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mirror.landmarks import (
    INDEX_MCP,
    INDEX_TIP,
    LANDMARK_NAMES,
    MIDDLE_MCP,
    N_LANDMARKS,
    PALM,
    PINKY_MCP,
    RING_MCP,
    THUMB_TIP,
    WRIST,
    Frame,
    HandLandmarks,
)


# --- the topology -----------------------------------------------------------
def test_there_are_twenty_one_landmarks() -> None:
    assert len(LANDMARK_NAMES) == N_LANDMARKS == 21


def test_named_indices_point_at_the_right_landmarks() -> None:
    """The regression for the bug the import-time guard caught on 2026-09-16.

    The first draft of LANDMARK_NAMES omitted RING_FINGER_TIP, which shifted
    PINKY_MCP from 17 to 16. PINKY_MCP is one of the three points Phase 2 builds
    the hand frame from, so an off-by-one there would have silently rotated every
    commanded pose.
    """
    assert LANDMARK_NAMES[WRIST] == "WRIST"
    assert LANDMARK_NAMES[THUMB_TIP] == "THUMB_TIP"
    assert LANDMARK_NAMES[INDEX_MCP] == "INDEX_FINGER_MCP"
    assert LANDMARK_NAMES[INDEX_TIP] == "INDEX_FINGER_TIP"
    assert LANDMARK_NAMES[MIDDLE_MCP] == "MIDDLE_FINGER_MCP"
    assert LANDMARK_NAMES[RING_MCP] == "RING_FINGER_MCP"
    assert LANDMARK_NAMES[PINKY_MCP] == "PINKY_MCP"


def test_palm_is_the_four_knuckles() -> None:
    assert PALM == (INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)
    assert all(LANDMARK_NAMES[i].endswith("_MCP") for i in PALM)
    assert WRIST not in PALM, "the wrist is a joint; it moves relative to the knuckles"


# --- validation -------------------------------------------------------------
def _valid(**overrides: object) -> HandLandmarks:
    base = {
        "image": np.zeros((21, 3)),
        "world": np.zeros((21, 3)),
        "handedness": "Right",
        "score": 0.9,
        "timestamp_ms": 0,
    }
    return HandLandmarks(**{**base, **overrides})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"image": np.zeros((20, 3))}, "image must be"),
        ({"world": np.zeros((21, 2))}, "world must be"),
        ({"image": np.full((21, 3), np.nan)}, "image contains non-finite"),
        ({"world": np.full((21, 3), np.inf)}, "world contains non-finite"),
        ({"handedness": "Middle"}, "handedness must be"),
        ({"score": 1.5}, r"score must be in \[0,1\]"),
        ({"score": -0.1}, r"score must be in \[0,1\]"),
    ],
)
def test_malformed_landmarks_are_rejected(overrides: dict[str, object], message: str) -> None:
    """A NaN landmark reaching the IK is a moving robot with no idea where it is."""
    with pytest.raises(ValueError, match=message):
        _valid(**overrides)


# --- derived quantities, against the real fixture ---------------------------
def test_fixture_is_self_describing(landmarks_golden: dict[str, object]) -> None:
    g = landmarks_golden
    assert tuple(g["landmark_names"]) == LANDMARK_NAMES  # type: ignore[arg-type]
    assert str(g["mediapipe_version"]) == "0.10.35"
    assert np.asarray(g["image"]).shape == (2, 21, 3)
    assert np.asarray(g["world"]).shape == (2, 21, 3)
    assert sorted(str(h) for h in g["handedness"]) == ["Left", "Right"]  # type: ignore[union-attr]


def test_world_landmarks_are_metric(golden_hands: list[HandLandmarks]) -> None:
    """The sanity check that the world output means metres at all.

    An adult palm (wrist to middle knuckle) is about 9-11 cm. If this drifts
    outside that, the world output has stopped being metric and every downstream
    scale factor is wrong.
    """
    for hand in golden_hands:
        assert 0.07 < hand.palm_span_m < 0.13, f"{hand.handedness}: {hand.palm_span_m:.4f} m"


def test_world_origin_is_near_the_palm_centre(golden_hands: list[HandLandmarks]) -> None:
    """MediaPipe documents only an "approximate geometric center". Measured, the
    origin tracks the four-knuckle mean to within a centimetre, while the mean of
    all 21 landmarks sits 2.5-2.7 cm away. Phase 2 relies on the former.
    """
    for hand in golden_hands:
        knuckle_mean = float(np.linalg.norm(hand.palm_centre_world))
        all_points_mean = float(np.linalg.norm(hand.world.mean(axis=0)))
        assert knuckle_mean < 0.01, f"{hand.handedness}: knuckle mean {knuckle_mean:.4f} m"
        assert knuckle_mean < all_points_mean, "knuckle mean should beat the 21-point mean"


def test_image_z_origin_is_the_wrist(golden_hands: list[HandLandmarks]) -> None:
    """The docs say the normalized z origin is the wrist. Verify, do not trust."""
    for hand in golden_hands:
        assert abs(hand.image[WRIST, 2]) < 1e-5, f"{hand.handedness}: {hand.image[WRIST, 2]:.3e}"


def test_image_xy_are_normalized(golden_hands: list[HandLandmarks]) -> None:
    for hand in golden_hands:
        assert np.all((hand.image[:, :2] >= -0.1) & (hand.image[:, :2] <= 1.1))


def test_pinch_ratio_distinguishes_the_two_hands(golden_hands: list[HandLandmarks]) -> None:
    """The clutch signal has to actually separate an open hand from a pinched one."""
    ratios = {h.handedness: h.pinch_ratio for h in golden_hands}
    assert ratios["Right"] < ratios["Left"], "the right hand is the more closed one"
    assert all(0.0 < r < 2.0 for r in ratios.values())


# --- the property that justifies ADR-002 ------------------------------------
@given(scale=st.floats(min_value=0.6, max_value=1.8))
@settings(max_examples=100, deadline=None)
def test_pinch_ratio_is_invariant_to_hand_size(
    golden_hands: list[HandLandmarks], scale: float
) -> None:
    """Scaling a whole hand must not change the clutch signal.

    This is the test that makes ADR-002's "no per-operator calibration" claim
    checkable rather than aspirational. `pinch_m` scales with the hand;
    `pinch_ratio` divides it out, so a hand 1.8x bigger reads the same.
    """
    for hand in golden_hands:
        bigger = HandLandmarks(
            image=hand.image,
            world=hand.world * scale,
            handedness=hand.handedness,
            score=hand.score,
            timestamp_ms=hand.timestamp_ms,
        )
        assert bigger.pinch_m == pytest.approx(hand.pinch_m * scale, rel=1e-12)
        assert bigger.pinch_ratio == pytest.approx(hand.pinch_ratio, rel=1e-12)


def test_pinch_ratio_of_a_degenerate_hand_is_infinite() -> None:
    """All 21 points collapsed to a single spot. Return inf, never divide by zero."""
    assert _valid().pinch_ratio == float("inf")


# --- Frame ------------------------------------------------------------------
def test_empty_frame_returns_none_rather_than_raising() -> None:
    """A hand leaving the frame is ordinary. The safety layer handles it."""
    frame = Frame(hands=(), timestamp_ms=0, latency_ms=1.0)
    assert frame.best() is None
    assert frame.best("Left") is None


def test_best_picks_the_highest_score() -> None:
    low = _valid(score=0.3, handedness="Left")
    high = _valid(score=0.9, handedness="Left")
    other = _valid(score=0.99, handedness="Right")
    frame = Frame(hands=(low, high, other), timestamp_ms=0, latency_ms=1.0)
    assert frame.best() is other
    assert frame.best("Left") is high
    assert frame.best("Right") is other


def test_frame_is_immutable() -> None:
    frame = Frame(hands=(), timestamp_ms=0, latency_ms=1.0)
    with pytest.raises((AttributeError, TypeError)):
        frame.timestamp_ms = 5  # type: ignore[misc]
