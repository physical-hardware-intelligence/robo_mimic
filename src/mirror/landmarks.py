"""MediaPipe hand landmarks, wrapped so the rest of the package never sees MediaPipe.

WHAT MEDIAPIPE ACTUALLY GIVES YOU
---------------------------------
Two coordinate systems per hand, 21 points each, and they are NOT the same data
in different units. Measured on the reference image (2026-09-16):

    world/image pair-distance ratio spread: 2.83x and 3.72x

A pure rescale would give a spread of exactly 1.00x. It does not, so the world
output is a genuinely separate 3-D reconstruction rather than a scaled
projection. That is why orientation taken from `world` is trustworthy.

    image  (21,3)  x,y in [0,1] of the frame; z is depth RELATIVE TO THE WRIST,
                   "roughly the same scale as x". Measured wrist z: -3.3e-07.
                   Use for: where the hand is ON SCREEN. Never for metric depth.

    world  (21,3)  metres. Origin is NOT the mean of the 21 points (that sits
                   2.5-2.7 cm away). Measured, the origin tracks the mean of the
                   four finger MCP knuckles (5, 9, 13, 17) to 0.44 cm / 0.75 cm
                   -- the PALM CENTRE. Good news: knuckles barely move relative
                   to each other, so this origin is stable under finger motion.
                   Use for: hand SHAPE and ORIENTATION. Never for position.

Neither origin is the camera, so neither tells you how far away the hand is.
That is the whole reason for ADR-002 (position is incremental, not absolute).

VERSION PINNING IS NOT OPTIONAL
-------------------------------
mediapipe 1.0.1 CRASHES on macOS arm64 inside its own graph:

    graph_service.h:139] Check failed: service_ Service is unavailable.
    -[DrishtiMetalHelper initWithCalculatorContext:]
    mediapipe::api2::TensorsToDetectionsCalculator::Open()

`TensorsToDetectionsCalculator` is the palm-detector stage, and it initialises
the Metal helper unconditionally -- passing `delegate=CPU` does not avoid it.
0.10.35 works. The pin in pyproject.toml is load-bearing.

DETERMINISM
-----------
Within a pinned version on one machine, `detect()` is BIT-IDENTICAL: 0.000e+00
deviation over 30 repeat calls and over 5 freshly constructed detectors. So the
golden fixtures can assert exact equality rather than a tolerance. Across
versions nothing is guaranteed -- 1.0.1 does not even run.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, Literal, Self

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from collections.abc import Iterator

F64 = npt.NDArray[np.float64]

#: MediaPipe's fixed 21-point topology, in index order. Mirrors
#: mediapipe.tasks.python.vision.hand_landmarker.HandLandmark, restated here so
#: that importing this module does not require mediapipe to be installed.
LANDMARK_NAMES: tuple[str, ...] = (
    "WRIST",
    "THUMB_CMC",
    "THUMB_MCP",
    "THUMB_IP",
    "THUMB_TIP",
    "INDEX_FINGER_MCP",
    "INDEX_FINGER_PIP",
    "INDEX_FINGER_DIP",
    "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP",
    "MIDDLE_FINGER_PIP",
    "MIDDLE_FINGER_DIP",
    "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP",
    "RING_FINGER_PIP",
    "RING_FINGER_DIP",
    "RING_FINGER_TIP",
    "PINKY_MCP",
    "PINKY_PIP",
    "PINKY_DIP",
    "PINKY_TIP",
)
N_LANDMARKS = 21

WRIST = 0
INDEX_MCP = 5
MIDDLE_MCP = 9
RING_MCP = 13
PINKY_MCP = 17
THUMB_TIP = 4
INDEX_TIP = 8

#: The four knuckles. Nearly rigid relative to one another, which is why Phase 2
#: builds the hand frame from these rather than from anything with a joint in it.
PALM: tuple[int, ...] = (INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)

Handedness = Literal["Left", "Right"]

# `LANDMARK_NAMES` is restated rather than imported, so it is checked here and
# again, against MediaPipe's real enum, in the `perception`-marked test suite.
# This guard has already earned its place: the first draft of the tuple was
# missing RING_FINGER_TIP, which silently shifted PINKY_MCP from 17 to 16 --
# and PINKY_MCP is one of the three points Phase 2 builds the hand frame from.
# A `raise`, not an `assert`, because `python -O` strips asserts.
if len(LANDMARK_NAMES) != N_LANDMARKS:  # pragma: no cover - import-time invariant
    raise RuntimeError(f"LANDMARK_NAMES has {len(LANDMARK_NAMES)}, expected {N_LANDMARKS}")
for _index, _name in ((WRIST, "WRIST"), (INDEX_MCP, "INDEX_FINGER_MCP"),
                      (MIDDLE_MCP, "MIDDLE_FINGER_MCP"), (RING_MCP, "RING_FINGER_MCP"),
                      (PINKY_MCP, "PINKY_MCP"), (THUMB_TIP, "THUMB_TIP"),
                      (INDEX_TIP, "INDEX_FINGER_TIP")):
    if LANDMARK_NAMES[_index] != _name:  # pragma: no cover - import-time invariant
        raise RuntimeError(f"index {_index} is {LANDMARK_NAMES[_index]}, expected {_name}")


@dataclass(frozen=True, slots=True)
class HandLandmarks:
    """One hand in one frame.

    `image` and `world` are both (21, 3) but mean different things -- see the
    module docstring. Confusing them is the single easiest way to build a
    plausible-looking teleoperator that does not work.
    """

    image: F64
    world: F64
    handedness: str
    score: float
    timestamp_ms: int

    def __post_init__(self) -> None:
        for name, value in (("image", self.image), ("world", self.world)):
            if value.shape != (N_LANDMARKS, 3):
                raise ValueError(f"{name} must be (21,3), got {value.shape}")
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} contains non-finite values")
        if self.handedness not in ("Left", "Right"):
            raise ValueError(f"handedness must be Left or Right, got {self.handedness!r}")
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"score must be in [0,1], got {self.score}")

    @property
    def palm_centre_world(self) -> F64:
        """Mean of the four MCP knuckles, in metres.

        Empirically within ~0.5-0.8 cm of MediaPipe's own world origin, so this
        is very nearly the zero vector. Computed anyway rather than assumed: the
        docs only promise an "approximate geometric center".
        """
        return np.asarray(self.world[list(PALM)].mean(axis=0), dtype=np.float64)

    @property
    def palm_span_m(self) -> float:
        """Wrist to middle knuckle, in metres. A real adult palm is 0.09-0.11 m.

        This is the sanity check that tells you whether the world output is
        metric at all. It is also the natural scale for normalising hand motion,
        because it cancels out how big the operator's hand is.
        """
        return float(np.linalg.norm(self.world[WRIST] - self.world[MIDDLE_MCP]))

    @property
    def pinch_m(self) -> float:
        """Thumb tip to index tip, in metres. The clutch signal for ADR-002."""
        return float(np.linalg.norm(self.world[THUMB_TIP] - self.world[INDEX_TIP]))

    @property
    def pinch_ratio(self) -> float:
        """`pinch_m` divided by palm span: a hand-size-independent clutch signal.

        Dividing by the palm span is the point. An absolute threshold in metres
        would need calibrating per operator; a ratio does not.
        """
        span = self.palm_span_m
        return self.pinch_m / span if span > 1e-9 else float("inf")


@dataclass(frozen=True, slots=True)
class Frame:
    """Everything detected in one frame: zero, one, or several hands."""

    hands: tuple[HandLandmarks, ...]
    timestamp_ms: int
    latency_ms: float

    def best(self, handedness: str | None = None) -> HandLandmarks | None:
        """Highest-confidence hand, optionally filtered by side. None if absent.

        Returning None rather than raising is deliberate: a hand leaving frame is
        ordinary, not exceptional, and the safety layer has a defined response to
        it. An exception here would turn a routine event into a control failure.
        """
        candidates = [h for h in self.hands if handedness is None or h.handedness == handedness]
        return max(candidates, key=lambda h: h.score) if candidates else None


class HandTracker:
    """Wraps MediaPipe's HandLandmarker. The only impure thing in the pipeline.

    mediapipe is imported INSIDE `__init__`, not at module scope, so that
    `import mirror.landmarks` works without it. That keeps the pure-math suite
    -- the one that runs in CI -- free of a 60 MB native dependency.

    Running modes:
        "image"  each frame independent. Use for stills and for fixtures you
                 want frame-order-independent.
        "video"  STATEFUL. MediaPipe tracks across frames, so frame N's result
                 depends on frames 0..N-1. Cheaper and smoother, but it means a
                 clip must always be replayed from the start to reproduce.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        num_hands: int = 1,
        mode: Literal["image", "video"] = "image",
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"hand landmarker model not found at {model_path}. Run `make assets`."
            )
        import mediapipe  # noqa: PLC0415
        from mediapipe.tasks.python import BaseOptions, vision  # noqa: PLC0415

        self._mp_image_cls = mediapipe.Image
        self._mp_format = mediapipe.ImageFormat
        self._mode = mode
        running = vision.RunningMode.VIDEO if mode == "video" else vision.RunningMode.IMAGE
        self._detector = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(model_path)),
                running_mode=running,
                num_hands=num_hands,
                min_hand_detection_confidence=min_detection_confidence,
                min_hand_presence_confidence=min_presence_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
        )

    def detect(self, rgb: npt.NDArray[np.uint8], timestamp_ms: int = 0) -> Frame:
        """Detect hands in one RGB frame.

        `rgb` must be uint8 HxWx3 in **RGB** order. OpenCV hands you BGR, so if
        you read with cv2 you must flip it. Feeding BGR does not error -- it
        quietly degrades detection, which is worse.
        """
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"expected HxWx3 RGB, got {rgb.shape}")
        if rgb.dtype != np.uint8:
            raise ValueError(f"expected uint8, got {rgb.dtype}")

        image = self._mp_image_cls(image_format=self._mp_format.SRGB, data=rgb)
        start = time.perf_counter()
        if self._mode == "video":
            result = self._detector.detect_for_video(image, timestamp_ms)
        else:
            result = self._detector.detect(image)
        latency_ms = (time.perf_counter() - start) * 1000.0
        return _to_frame(result, timestamp_ms, latency_ms)

    def close(self) -> None:
        self._detector.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def _to_frame(result: Any, timestamp_ms: int, latency_ms: float) -> Frame:
    """MediaPipe's result objects -> our frozen, validated dataclasses.

    Done eagerly rather than lazily so that a malformed detection fails HERE,
    with the frame in hand, instead of three stages downstream.
    """
    hands = []
    for index in range(len(result.hand_landmarks)):
        category = result.handedness[index][0]
        hands.append(
            HandLandmarks(
                image=_xyz(result.hand_landmarks[index]),
                world=_xyz(result.hand_world_landmarks[index]),
                handedness=str(category.category_name),
                score=float(category.score),
                timestamp_ms=timestamp_ms,
            )
        )
    return Frame(hands=tuple(hands), timestamp_ms=timestamp_ms, latency_ms=latency_ms)


def _xyz(points: Any) -> F64:
    return np.array([[p.x, p.y, p.z] for p in points], dtype=np.float64)


def iter_video_rgb(path: str | Path) -> Iterator[tuple[int, npt.NDArray[np.uint8]]]:
    """Yield (timestamp_ms, RGB frame) from a video file.

    Converts BGR->RGB here, once, so no caller has to remember. Timestamps come
    from the container rather than a frame counter, because MediaPipe's video
    mode wants real monotonic milliseconds.
    """
    import cv2  # noqa: PLC0415

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"cannot open video {path}")
    try:
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            timestamp_ms = int(capture.get(cv2.CAP_PROP_POS_MSEC))
            yield timestamp_ms, np.ascontiguousarray(bgr[:, :, ::-1], dtype=np.uint8)
    finally:
        capture.release()


def read_image_rgb(path: str | Path) -> npt.NDArray[np.uint8]:
    """Read a still image as uint8 RGB."""
    import cv2  # noqa: PLC0415

    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(f"cannot read image {path}")
    return np.ascontiguousarray(bgr[:, :, ::-1], dtype=np.uint8)
