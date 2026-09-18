#!/usr/bin/env python
"""Live hand tracking from the built-in camera, with an on-screen readout.

    make view                 look, record nothing
    make record               look, and press SPACE to capture a clip

KEYS
    SPACE   clutch on / off        <- the arm only moves while this is ON
    C       start / stop capturing a clip
    R       reset the clutch and the anchor
    Q/ESC   quit

WHAT YOU SHOULD SEE
    green skeleton          21 landmarks
    red/green/blue axes     the palm frame: x, y (fingers), z (out of palm)
    CLUTCH                  grey = off (nothing moves), green = on
    JAW bar                 gripper opening. Pinch to close, spread to open
    PINCH                   the raw thumb-index ratio behind the jaw
    TOOL                    where the arm would be commanded, in metres

DEADMAN CAVEAT
SPACE here is a TOGGLE, not a held key. OpenCV cannot detect key-hold: macOS
sends one keydown, then nothing for ~500 ms, then repeats. A "held" heuristic
would drop out during that gap, which is worse than a toggle.

`Retargeter.step` takes `engage` as a plain boolean precisely so the source can
change without touching the logic. Phase 7 on real hardware needs a genuine
momentary switch -- a footswitch or gamepad trigger -- not this toggle.

The preview is mirrored so it behaves like a mirror. MediaPipe assumes an
unmirrored image, so it labels your right hand "Left". That is expected and
harmless: the frame construction handles both, consistently.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from robo_mimic.handframe import DegenerateHandError, hand_pose  # noqa: E402
from robo_mimic.landmarks import HandTracker  # noqa: E402
from robo_mimic.retarget import Clutch, RetargetConfig, Retargeter  # noqa: E402
from robo_mimic.types import Pose  # noqa: E402

MODEL = ROOT / "assets" / "hand_landmarker.task"
CLIPS = ROOT / "fixtures" / "clips"

BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]

WHITE = (255, 255, 255)
GREY = (150, 150, 150)
GREEN = (80, 230, 120)
RED = (60, 60, 240)
BLUE = (240, 160, 60)
AMBER = (0, 190, 255)


def _fourcc(code: str) -> int:
    """cv2.VideoWriter_fourcc: present at runtime, absent from the cv2 stubs.

    Imports cv2 itself because these scripts import it lazily inside their
    functions, so a module-level reference would not resolve.
    """
    import cv2

    return int(cv2.VideoWriter_fourcc(*code))  # type: ignore[attr-defined]


def draw(
    canvas: Any, hand: Any, pose: Any, command: Any,
    config: Any, stats: dict[str, Any],
) -> Any:
    import cv2

    height, width = canvas.shape[:2]

    if hand is not None:
        pixels = (hand.image[:, :2] * [width, height]).astype(int)
        for a, b in BONES:
            cv2.line(canvas, tuple(pixels[a]), tuple(pixels[b]), GREEN, 2)
        for x, y in pixels:
            cv2.circle(canvas, (x, y), 3, WHITE, -1)

        if pose is not None:
            # Project the palm axes using the image-space palm scale, so the
            # drawn axes shrink and grow with the hand instead of being fixed.
            origin = pixels[[5, 9, 13, 17]].mean(axis=0).astype(int)
            length = float(np.linalg.norm(pixels[9] - pixels[0])) * 0.8
            for k, colour in ((0, RED), (1, GREEN), (2, BLUE)):
                axis = pose.rotation[:, k]
                tip = origin + (axis[:2] * [1, 1] * length).astype(int)
                cv2.arrowedLine(canvas, tuple(origin), tuple(tip), colour, 3, tipLength=0.25)

    panel = canvas[0:156, 0:430]
    canvas[0:156, 0:430] = (panel * 0.35).astype(np.uint8)

    def text(row: int, label: str, value: str,
             colour: tuple[int, int, int] = WHITE) -> None:
        cv2.putText(canvas, label, (12, row), cv2.FONT_HERSHEY_SIMPLEX, 0.45, GREY, 1,
                    cv2.LINE_AA)
        cv2.putText(canvas, value, (118, row), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1,
                    cv2.LINE_AA)

    def bar(row: int, value: float, colour: tuple[int, int, int]) -> int:
        x0, width_px = 118, 180
        cv2.rectangle(canvas, (x0, row - 12), (x0 + width_px, row), (60, 60, 60), -1)
        filled = int(np.clip(value, 0, 1) * width_px)
        cv2.rectangle(canvas, (x0, row - 12), (x0 + filled, row), colour, -1)
        return x0 + width_px + 8

    text(24, "fps / mp", f"{stats['fps']:5.1f}  {stats['latency']:5.1f} ms")

    engaged = command.clutch is Clutch.ENGAGED
    text(48, "clutch", f"{'ON' if engaged else 'off'}  ({command.reason})",
         GREEN if engaged else GREY)

    if command.gripper is None:
        text(72, "jaw", "-- gated by the clutch --", GREY)
    else:
        end = bar(72, command.gripper, GREEN)
        cv2.putText(canvas, f"{command.gripper:.2f}", (end, 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1, cv2.LINE_AA)
        cv2.putText(canvas, "jaw", (12, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.45, GREY, 1,
                    cv2.LINE_AA)

    ratio = command.pinch_ratio
    if ratio is None:
        text(96, "pinch", "--", GREY)
    else:
        end = bar(96, ratio / 1.2, AMBER)
        for threshold in (config.pinch_closed, config.pinch_open):
            tick = 118 + int(np.clip(threshold / 1.2, 0, 1) * 180)
            cv2.line(canvas, (tick, 80), (tick, 98), WHITE, 1)
        cv2.putText(canvas, f"{ratio:.2f}", (end, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    WHITE, 1, cv2.LINE_AA)
        cv2.putText(canvas, "pinch", (12, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.45, GREY, 1,
                    cv2.LINE_AA)

    if command.target is not None:
        p = command.target.position * 100
        text(122, "tool cm", f"x {p[0]:+6.1f}  y {p[1]:+6.1f}  z {p[2]:+6.1f}", GREEN)
    else:
        text(122, "tool cm", "holding", GREY)

    if stats["recording"]:
        cv2.circle(canvas, (width - 28, 28), 10, RED, -1)
        cv2.putText(canvas, f"REC {stats['frames']}", (width - 140, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 2, cv2.LINE_AA)
    cv2.putText(canvas, "SPACE clutch   C capture   R reset   Q quit", (12, height - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, GREY, 1, cv2.LINE_AA)
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="device index (0 = built-in)")
    parser.add_argument("--video", help="replay a recorded clip instead of the camera")
    parser.add_argument("--name", default="clip", help="base name for recordings")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--no-flip", action="store_true", help="disable the mirrored preview")
    parser.add_argument("--headless", action="store_true", help="no window; record N frames")
    parser.add_argument("--frames", type=int, default=0, help="headless: how many frames")
    args = parser.parse_args()

    if not MODEL.exists():
        print("model missing. Run `make assets`.", file=sys.stderr)
        return 1

    import cv2

    CLIPS.mkdir(parents=True, exist_ok=True)
    replaying = args.video is not None
    if replaying:
        camera = cv2.VideoCapture(args.video)
        if not camera.isOpened():
            print(f"cannot open {args.video}", file=sys.stderr)
            return 1
    else:
        camera = cv2.VideoCapture(args.camera)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        if not camera.isOpened():
            print(
                f"cannot open camera {args.camera}.\n"
                "On macOS the terminal app needs Camera permission:\n"
                "  System Settings > Privacy & Security > Camera",
                file=sys.stderr,
            )
            return 1

    ok, probe = camera.read()
    if not ok:
        print("camera opened but returned no frame.", file=sys.stderr)
        camera.release()
        return 1
    height, width = probe.shape[:2]
    aspect = height / width
    source = args.video if replaying else f"camera {args.camera}"
    print(f"{source}: {width}x{height}  aspect {aspect:.3f}")

    config = RetargetConfig(aspect=aspect)
    retargeter = Retargeter(config)
    tool_home = Pose(position=np.array([0.20, 0.0, 0.15]), rotation=np.eye(3))

    engaged = False
    recording = False
    writer = None
    captured: list[dict[str, object]] = []
    smoothed_fps = 0.0
    previous = time.perf_counter()

    with HandTracker(MODEL, num_hands=1, mode="video") as tracker:
        frame_index = 0
        while True:
            ok, bgr = camera.read()
            if not ok:
                break
            # A recorded clip was already mirrored when it was captured.
            if not args.no_flip and not replaying:
                bgr = cv2.flip(bgr, 1)
            rgb = np.ascontiguousarray(bgr[:, :, ::-1], dtype=np.uint8)

            timestamp_ms = int(time.perf_counter() * 1000)
            detection = tracker.detect(rgb, timestamp_ms)
            hand = detection.best()

            pose = None
            if hand is not None:
                try:
                    pose = hand_pose(hand)
                except DegenerateHandError:
                    pose = None
            command = retargeter.step(hand, tool_home, timestamp_ms, engage=engaged)

            now = time.perf_counter()
            instant = 1.0 / max(now - previous, 1e-6)
            previous = now
            smoothed_fps = instant if smoothed_fps == 0 else 0.9 * smoothed_fps + 0.1 * instant

            if recording:
                captured.append(
                    {
                        "t": timestamp_ms,
                        "image": hand.image if hand is not None else None,
                        "world": hand.world if hand is not None else None,
                        "handedness": hand.handedness if hand is not None else "",
                        "score": hand.score if hand is not None else 0.0,
                    }
                )
                if writer is not None:
                    writer.write(bgr)

            canvas = draw(
                bgr.copy(), hand, pose, command, config,
                {"fps": smoothed_fps, "latency": detection.latency_ms,
                 "recording": recording, "frames": len(captured)},
            )

            frame_index += 1
            if args.headless:
                if args.frames and frame_index >= args.frames:
                    break
                continue

            cv2.imshow("robo_mimic", canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                retargeter.reset()
                engaged = False
            if key == ord(" "):
                engaged = not engaged
            if key == ord("c"):
                recording = not recording
                if recording:
                    captured = []
                    stamp = time.strftime("%Y%m%d-%H%M%S")
                    path = CLIPS / f"{args.name}-{stamp}.mp4"
                    writer = cv2.VideoWriter(
                        str(path), _fourcc("mp4v"), 30.0, (width, height)
                    )
                    print(f"recording -> {path.name}")
                else:
                    if writer is not None:
                        writer.release()
                        writer = None
                    save(captured, args.name, aspect, width, height)

    if writer is not None:
        writer.release()
    camera.release()
    if not args.headless:
        cv2.destroyAllWindows()
    return 0


def save(
    captured: list[dict[str, object]], name: str, aspect: float, width: int, height: int
) -> None:
    """Landmark stream -> npz. Missing frames are kept, with a `seen` flag.

    Dropouts are data, not gaps to delete: the grace-period logic exists because
    of them, so a clip that never drops a frame cannot test it.
    """
    if not captured:
        print("nothing captured")
        return
    count = len(captured)
    image = np.zeros((count, 21, 3))
    world = np.zeros((count, 21, 3))
    seen = np.zeros(count, dtype=bool)
    for index, row in enumerate(captured):
        if row["image"] is not None:
            image[index] = np.asarray(row["image"])
            world[index] = np.asarray(row["world"])
            seen[index] = True
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = CLIPS / f"{name}-{stamp}.npz"
    np.savez_compressed(
        path,
        image=image, world=world, seen=seen,
        t_ms=np.array([r["t"] for r in captured]),
        handedness=np.array([r["handedness"] for r in captured]),
        score=np.array([r["score"] for r in captured]),
        aspect=np.array(aspect), width=np.array(width), height=np.array(height),
    )
    print(
        f"saved {path.name}: {count} frames, {int(seen.sum())} with a hand "
        f"({100 * seen.mean():.0f}%), {path.stat().st_size / 1024:.0f} KB"
    )


if __name__ == "__main__":
    raise SystemExit(main())
