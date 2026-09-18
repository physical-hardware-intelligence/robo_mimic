"""Tests that genuinely need mediapipe and the fetched assets.

Everything here is marked `perception`, so `make test` skips it. These are the
tests that pin our restated constants to MediaPipe's real ones, prove the golden
fixture still reproduces, and hold the latency budget.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from robo_mimic.landmarks import LANDMARK_NAMES, HandTracker, read_image_rgb

from .conftest import MODEL, REFERENCE_PNG

pytestmark = pytest.mark.perception

#: Budget, from the Phase 1 measurement (p90 = 11.76 ms on an M-series CPU).
#: Set at 3x so a busy or slower machine does not fail the build, while a real
#: regression -- a model swap, a resolution change, losing the CPU delegate --
#: moves the number by far more than 3x.
LATENCY_BUDGET_MS = 35.0

#: Within a pinned version on one machine, detect() is bit-identical. Measured:
#: 0.000e+00 over 30 repeat calls and over 5 freshly built detectors.
EXACT = 0.0


@pytest.fixture(autouse=True)
def _require_assets() -> None:
    if not MODEL.exists() or not REFERENCE_PNG.exists():
        pytest.skip("assets not fetched -- run `make assets && make fixtures`")


@pytest.fixture(scope="module")
def tracker() -> Any:
    with HandTracker(MODEL, num_hands=2, mode="image") as handle:
        yield handle


def test_our_landmark_names_match_mediapipes_enum() -> None:
    """We restate the topology so the module imports without mediapipe. Pin it."""
    from mediapipe.tasks.python.vision.hand_landmarker import HandLandmark

    assert LANDMARK_NAMES == tuple(e.name for e in HandLandmark)


def test_the_pinned_version_is_the_one_installed() -> None:
    """1.0.1 aborts on macOS arm64 inside TensorsToDetectionsCalculator. The pin
    is load-bearing, and the golden landmarks are version-specific.
    """
    import mediapipe

    assert mediapipe.__version__ == "0.10.35"


def test_png_decodes_identically_for_everyone(landmarks_golden: dict[str, Any]) -> None:
    """The fixture is only reproducible if the pixels are.

    The source JPEG is NOT safe here: decoding it with MediaPipe's loader versus
    OpenCV's differs by up to 3 levels on 2.79 percent of pixels, which moves
    world landmarks by 0.7370 mm. PNG is lossless, so every decoder agrees and
    the landmarks match to 0.000000 mm. This test guards that property.
    """
    import hashlib

    rgb = read_image_rgb(REFERENCE_PNG)
    assert hashlib.sha256(rgb.tobytes()).hexdigest() == str(
        landmarks_golden["decoded_pixels_sha256"]
    )
    assert tuple(rgb.shape) == tuple(landmarks_golden["image_shape"])


@pytest.mark.golden
def test_detection_reproduces_the_golden_exactly(
    tracker: Any, landmarks_golden: dict[str, Any]
) -> None:
    """The core Phase 1 gate: the same pixels give back the same landmarks.

    Marked `golden` and EXCLUDED FROM CI on purpose. The fixture was generated on
    macOS arm64 with mediapipe 0.10.35. Bit-identity has been verified there --
    across repeat calls and across freshly built detectors -- but NOT across
    operating systems or architectures, where XNNPACK may take different kernel
    paths. Asserting exact equality on a Linux runner would be a claim we have
    not tested. Regenerate the fixture per platform, or relax to a tolerance,
    before turning this on in CI.
    """
    frame = tracker.detect(read_image_rgb(REFERENCE_PNG))
    hands = sorted(frame.hands, key=lambda h: h.handedness)
    assert len(hands) == 2

    for index, hand in enumerate(hands):
        assert hand.handedness == str(landmarks_golden["handedness"][index])
        assert np.abs(hand.image - landmarks_golden["image"][index]).max() == EXACT
        assert np.abs(hand.world - landmarks_golden["world"][index]).max() == EXACT
        assert hand.score == pytest.approx(float(landmarks_golden["score"][index]), abs=1e-12)


def test_detection_is_bit_identical_across_repeat_calls(tracker: Any) -> None:
    rgb = read_image_rgb(REFERENCE_PNG)
    first = sorted(tracker.detect(rgb).hands, key=lambda h: h.handedness)
    for _ in range(10):
        again = sorted(tracker.detect(rgb).hands, key=lambda h: h.handedness)
        for a, b in zip(first, again, strict=True):
            assert np.abs(a.world - b.world).max() == EXACT
            assert np.abs(a.image - b.image).max() == EXACT


def test_detection_is_bit_identical_across_fresh_detectors() -> None:
    """A rebuilt detector must agree, or fixtures could not survive a restart."""
    rgb = read_image_rgb(REFERENCE_PNG)
    reference = None
    for _ in range(3):
        with HandTracker(MODEL, num_hands=2, mode="image") as handle:
            hands = sorted(handle.detect(rgb).hands, key=lambda h: h.handedness)
        current = np.stack([h.world for h in hands])
        if reference is None:
            reference = current
        else:
            assert np.abs(current - reference).max() == EXACT


def test_latency_is_within_budget(tracker: Any) -> None:
    """Measured p90 was 11.76 ms -> 85 FPS. Budget is 3x that."""
    import time

    rgb = read_image_rgb(REFERENCE_PNG)
    tracker.detect(rgb)  # warm up
    timings = []
    for _ in range(30):
        start = time.perf_counter()
        tracker.detect(rgb)
        timings.append((time.perf_counter() - start) * 1000.0)
    p90 = float(np.percentile(timings, 90))
    assert p90 < LATENCY_BUDGET_MS, f"p90 {p90:.2f} ms exceeds {LATENCY_BUDGET_MS} ms"


def test_an_empty_scene_yields_no_hands(tracker: Any) -> None:
    """Mid-grey, no hand. Must return an empty Frame, not raise, not hallucinate."""
    blank = np.full((480, 640, 3), 128, dtype=np.uint8)
    frame = tracker.detect(blank)
    assert frame.hands == ()
    assert frame.best() is None


@pytest.mark.parametrize(
    ("array", "message"),
    [
        (np.zeros((480, 640), dtype=np.uint8), "expected HxWx3"),
        (np.zeros((480, 640, 4), dtype=np.uint8), "expected HxWx3"),
        (np.zeros((480, 640, 3), dtype=np.float32), "expected uint8"),
    ],
)
def test_malformed_input_is_rejected_loudly(
    tracker: Any, array: np.ndarray, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        tracker.detect(array)


def test_a_missing_model_says_so() -> None:
    with pytest.raises(FileNotFoundError, match="hand landmarker model not found"):
        HandTracker("assets/does_not_exist.task")


def test_bgr_input_is_silently_worse_which_is_why_we_convert_centrally(tracker: Any) -> None:
    """Feeding BGR does not error. It quietly degrades detection.

    This is the failure mode the `iter_video_rgb` / `read_image_rgb` helpers
    exist to prevent: the conversion happens in one place so no caller has to
    remember. The test documents the hazard rather than asserting a fixed
    degradation, which would be brittle.
    """
    rgb = read_image_rgb(REFERENCE_PNG)
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    correct = tracker.detect(rgb)
    swapped = tracker.detect(bgr)
    assert len(correct.hands) == 2
    if len(swapped.hands) == 2:
        drift = max(
            float(np.abs(a.world - b.world).max())
            for a, b in zip(
                sorted(correct.hands, key=lambda h: h.handedness),
                sorted(swapped.hands, key=lambda h: h.handedness),
                strict=True,
            )
        )
        assert drift > 1e-6, "BGR and RGB gave identical results -- the channel swap did nothing?"
