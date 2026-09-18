#!/usr/bin/env python
"""Live teleoperation: your hand on the left, the simulated arm on the right.

    make teleop                     camera -> sim, side by side
    make teleop ARGS="--bench"      no window; print the latency budget
    make teleop ARGS="--source synthetic"   no camera needed, for checking the loop
    make teleop ARGS="--source video --video fixtures/clips/x.mp4"

KEYS
    SPACE   clutch on / off        <- the arm only moves while this is ON
    R       reset the clutch and re-seed the limiter from the arm's pose
    Q/ESC   quit

THE PIPELINE, AND WHERE THE TIME GOES
    capture -> mediapipe -> hand frame -> retarget -> project -> safety
            -> sim.set_target -> sim.advance (16 x 2 ms) -> render

Every stage is timed separately and shown live, because a single FPS number
hides which stage is the problem. MEASURED IN REAL USE, from a live session
recording rather than a synthetic loop:

    capture (camera)                13.6 - 17.9 ms   waiting on the sensor
    mediapipe, VIDEO mode                  11.7 ms
    retarget + project + safety + sim  0.8 -  1.5 ms
    render + side-by-side                  ~13.0 ms
    TOTAL                           40.2 - 63.5 ms  ->  13 - 20 fps

A correction to an earlier claim of mine: I first measured MediaPipe at 3.87 ms
in video mode and reported a 3x saving over IMAGE mode. That benchmark fed the
SAME frame repeatedly, so tracking never lost lock and the palm detector never
re-ran. On 181 distinct moving frames from a real session:

    video mode   12.17 ms mean, p95 20.02,  hand found 173/181  (96 percent)
    image mode   23.98 ms mean, p95 31.37,  hand found  67/181  (37 percent)

So video mode is the right choice -- but for RELIABILITY far more than speed.
Image mode loses the hand on two thirds of moving frames. ~12 ms is the honest
cost, and it matches what the live overlay reports.

  RENDERING, not perception, is the bottleneck. Attributed precisely:

      arm.frame(renderer)   23.36 ms
      annotate()             0.65 ms
      side_by_side()         0.17 ms
      cv2.imshow contention ~11.90 ms   (window open vs not)

  Lowering the render resolution does NOT help -- 640x480 costs 25.78 ms and
  320x240 costs 22.87 ms, so this is MuJoCo's per-call scene setup and GL
  round-trip, not pixel fill. THROTTLING the redraw is the only lever that
  works, which is why `--render-every` defaults to 2: the physics still runs
  every frame, the sim view refreshes at ~15 Hz, and the loop keeps up with a
  30 fps camera instead of falling behind it and accumulating lag.

  (phi's own MuJoCo notes reached the same conclusion from the other direction:
  redraw every 8th step, because a 60 Hz display cannot show 500 redraws/s.)

WHY THREE SOURCES
`synthetic` drives the identical pipeline from the committed fixture, so the
loop can be verified and benchmarked on a machine with no camera permission --
which is exactly the situation the agent building this was in. `video` replays a
recording. `camera` is the real thing.
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

from mirror.landmarks import MIDDLE_MCP, PALM, WRIST, HandLandmarks, HandTracker  # noqa: E402
from mirror.project import project  # noqa: E402
from mirror.retarget import Command, RetargetConfig, Retargeter  # noqa: E402
from mirror.safety import SafetyConfig, SafetyLimiter  # noqa: E402
from mirror.sim import SimArm  # noqa: E402
from mirror.timing import Budget  # noqa: E402
from mirror.types import Pose  # noqa: E402

MODEL = ROOT / "assets" / "hand_landmarker.task"
SCENE = ROOT / "model" / "scene.xml"
GOLDEN = ROOT / "fixtures" / "landmarks_golden.npz"

START = {
    "shoulder_pan": 0.0,
    "shoulder_lift": -40.0,
    "elbow_flex": 55.0,
    "wrist_flex": -25.0,
    "wrist_roll": 0.0,
}

BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]
WHITE, GREY, GREEN, RED, AMBER = (
    (255, 255, 255), (150, 150, 150), (80, 230, 120), (60, 60, 240), (0, 190, 255)
)

#: Stages, in pipeline order. Named here so the readout and the bench agree.
STAGES = ("capture", "mediapipe", "retarget", "project", "safety", "sim", "render")


# --- frame sources ---------------------------------------------------------
class SyntheticSource:
    """The committed fixture's real landmarks, moved on a known path.

    Bypasses capture and MediaPipe entirely -- it yields `HandLandmarks`, not
    pixels -- so the loop can be benchmarked without a camera. The perception
    cost it omits was measured separately: 11.8 ms p90.
    """

    needs_detector = False

    def __init__(self) -> None:
        with np.load(GOLDEN) as data:
            self._base = HandLandmarks(
                image=data["image"][0], world=data["world"][0],
                handedness=str(data["handedness"][0]),
                score=float(data["score"][0]), timestamp_ms=0,
            )
        self._index = 0
        self.width, self.height = 640, 480

    def read(self) -> tuple[np.ndarray | None, HandLandmarks | None]:
        t = self._index / 31.0
        self._index += 1
        image = self._base.image.copy()
        centre = image[list(PALM), :2].mean(axis=0)
        image[:, :2] = centre + (image[:, :2] - centre) * (1 + 0.18 * np.sin(1.26 * t))
        image[:, 0] += 0.10 * np.sin(2.2 * t)
        image[:, 1] += -0.06 * np.sin(1.57 * t)
        world = self._base.world.copy()
        span = float(np.linalg.norm(world[WRIST] - world[MIDDLE_MCP]))
        pinch = 0.9 if (t % 4.0) < 2.0 else 0.12
        world[4] = world[8] + np.array([pinch * span, 0.0, 0.0])
        hand = HandLandmarks(
            image=image, world=world, handedness=self._base.handedness,
            score=self._base.score, timestamp_ms=int(t * 1000),
        )
        canvas = np.full((self.height, self.width, 3), 40, dtype=np.uint8)
        return canvas, hand

    def release(self) -> None:
        pass


class PixelSource:
    """Camera or video file. Yields BGR frames; MediaPipe does the rest."""

    needs_detector = True

    def __init__(self, camera: int, video: str | None, flip: bool) -> None:
        import cv2

        self._cv2 = cv2
        self._flip = flip and video is None
        if video is not None:
            self._cap = cv2.VideoCapture(video)
            if not self._cap.isOpened():
                raise FileNotFoundError(f"cannot open {video}")
        else:
            self._cap = cv2.VideoCapture(camera)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            # Ask for the shallowest queue the backend will give. `capture` was
            # measured at 13.6-17.9 ms in real use: most of that is waiting on
            # the sensor, which is unavoidable, but a deep queue adds latency on
            # top by handing back stale frames. macOS AVFoundation may ignore
            # this; setting it costs nothing when it does.
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not self._cap.isOpened():
                raise RuntimeError(
                    f"cannot open camera {camera}.\n"
                    "macOS grants camera access PER APPLICATION. Run this from the\n"
                    "terminal you normally use and click Allow, or enable it under\n"
                    "System Settings > Privacy & Security > Camera."
                )
        ok, probe = self._cap.read()
        if not ok:
            raise RuntimeError("source opened but returned no frame")
        self.height, self.width = probe.shape[:2]

    def read(self) -> tuple[np.ndarray | None, None]:
        ok, bgr = self._cap.read()
        if not ok:
            return None, None
        if self._flip:
            bgr = self._cv2.flip(bgr, 1)
        return bgr, None

    def release(self) -> None:
        self._cap.release()


# --- overlay ----------------------------------------------------------------
def annotate(
    bgr: np.ndarray, hand: HandLandmarks | None, command: Command, budget: Budget,
    fps: float, engaged: bool, held: bool,
) -> np.ndarray:
    import cv2

    height, width = bgr.shape[:2]
    if hand is not None:
        pixels = (hand.image[:, :2] * [width, height]).astype(int)
        for a, b in BONES:
            cv2.line(bgr, tuple(pixels[a]), tuple(pixels[b]), GREEN, 2)
        for x, y in pixels:
            cv2.circle(bgr, (x, y), 3, WHITE, -1)

    panel = bgr[0 : 30 + 18 * (len(STAGES) + 3), 0:300]
    bgr[0 : panel.shape[0], 0:300] = (panel * 0.3).astype(np.uint8)

    def line(row: int, label: str, value: str, colour: tuple[int, int, int] = WHITE) -> None:
        cv2.putText(bgr, label, (10, row), cv2.FONT_HERSHEY_SIMPLEX, 0.42, GREY, 1, cv2.LINE_AA)
        cv2.putText(bgr, value, (112, row), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)

    row = 22
    line(row, "fps", f"{fps:5.1f}   budget {budget.total_ms():5.1f} ms")
    row += 20
    state = "ON" if engaged else "off"
    line(row, "clutch", f"{state}  ({command.reason})", GREEN if engaged else GREY)
    row += 18
    jaw = "--" if command.gripper is None else f"{command.gripper:.2f}"
    line(row, "jaw", jaw, GREEN if command.gripper is not None else GREY)
    row += 18
    if command.target is not None:
        p = command.target.position * 100
        line(row, "tool cm", f"{p[0]:+5.1f} {p[1]:+5.1f} {p[2]:+5.1f}", GREEN)
    else:
        line(row, "tool cm", "holding" if held else "--", GREY)
    row += 22
    for row_stats in budget.table():
        colour = AMBER if row_stats.mean_ms > 8.0 else WHITE
        line(row, f"  {row_stats.stage}", f"{row_stats.mean_ms:6.2f} ms", colour)
        row += 18

    cv2.putText(bgr, "SPACE clutch   R reset   Q quit", (10, height - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, GREY, 1, cv2.LINE_AA)
    return bgr


def side_by_side(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    import cv2

    height = min(left.shape[0], right.shape[0])
    scale_l = height / left.shape[0]
    scale_r = height / right.shape[0]
    a = cv2.resize(left, (int(left.shape[1] * scale_l), height))
    b = cv2.resize(right, (int(right.shape[1] * scale_r), height))
    return np.hstack([a, b])


# --- the loop ---------------------------------------------------------------
def main() -> int:  # noqa: PLR0912, PLR0915
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("camera", "video", "synthetic"), default="camera")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--video")
    parser.add_argument("--no-flip", action="store_true")
    parser.add_argument("--bench", action="store_true", help="no window; print the budget")
    parser.add_argument("--frames", type=int, default=0, help="stop after N frames")
    parser.add_argument("--engage", action="store_true", help="start with the clutch ON")
    parser.add_argument(
        "--render-every", type=int, default=2, metavar="N",
        help="redraw the sim view every Nth frame (physics always runs). Default 2: "
        "rendering costs ~23 ms and is the bottleneck. Use 1 for a smoother sim "
        "view at a lower overall frame rate.",
    )
    args = parser.parse_args()

    if not SCENE.exists():
        print("model missing. Run `make model`.", file=sys.stderr)
        return 1

    try:
        source: Any = (
            SyntheticSource()
            if args.source == "synthetic"
            else PixelSource(args.camera, args.video if args.source == "video" else None,
                             not args.no_flip)
        )
    except (RuntimeError, FileNotFoundError) as error:
        print(f"\n{error}\n", file=sys.stderr)
        return 1

    detector = None
    if source.needs_detector:
        if not MODEL.exists():
            print("model missing. Run `make assets`.", file=sys.stderr)
            return 1
        detector = HandTracker(MODEL, num_hands=1, mode="video")

    aspect = source.height / source.width
    retargeter = Retargeter(RetargetConfig(aspect=aspect))
    limiter = SafetyLimiter(SafetyConfig())
    limiter.reset(START)
    arm = SimArm(SCENE, interpolate=True, start=START)
    renderer = arm.renderer()

    budget = Budget(STAGES)
    sim_view: np.ndarray | None = None
    frame: np.ndarray | None = None
    engaged = bool(args.engage)
    fps = 0.0
    previous = time.perf_counter()
    index = 0
    print(f"source {args.source}: {source.width}x{source.height}  aspect {aspect:.3f}")
    if not args.bench:
        print("SPACE = clutch. The arm only moves while it is ON.")

    while True:
        with budget.measure("capture"):
            bgr, supplied = source.read()
        if bgr is None:
            break

        hand = supplied
        with budget.measure("mediapipe"):
            if detector is not None:
                rgb = np.ascontiguousarray(bgr[:, :, ::-1], dtype=np.uint8)
                hand = detector.detect(rgb, int(time.perf_counter() * 1000)).best()

        with budget.measure("retarget"):
            pose_now = arm.pose()
            command = retargeter.step(
                hand, Pose(position=pose_now.position, rotation=pose_now.rotation),
                int(time.perf_counter() * 1000), engage=engaged,
            )

        joints = gripper = None
        with budget.measure("project"):
            if command.target is not None:
                joints = project(command.target, current=limiter.last).joints
                gripper = command.gripper

        now = time.perf_counter()
        dt = max(now - previous, 1e-4)
        previous = now
        with budget.measure("safety"):
            safe = limiter.step(joints, gripper, dt)

        with budget.measure("sim"):
            arm.set_target(safe.joints, safe.gripper)
            arm.advance(dt)

        with budget.measure("render"):
            if not args.bench:
                if index % max(1, args.render_every) == 0 or sim_view is None:
                    sim_view = np.ascontiguousarray(arm.frame(renderer)[:, :, ::-1])
                left = annotate(bgr.copy(), hand, command, budget,
                                fps, engaged, safe.held)
                frame = side_by_side(left, sim_view)

        instant = 1.0 / dt
        fps = instant if fps == 0 else 0.9 * fps + 0.1 * instant
        index += 1
        if args.frames and index >= args.frames:
            break
        if args.bench:
            continue

        import cv2

        cv2.imshow("mirror -- hand | sim", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            engaged = not engaged
        if key == ord("r"):
            retargeter.reset()
            limiter.reset(arm.joints)
            engaged = False

    source.release()
    arm.close()
    if not args.bench:
        import cv2

        cv2.destroyAllWindows()

    print(f"\n{index} frames, {fps:.1f} fps sustained")
    print(f"\n  {'stage':<12} {'mean':>9} {'p95':>9} {'max':>9}")
    for row_stats in budget.table():
        print(f"  {row_stats.stage:<12} {row_stats.mean_ms:>8.2f}ms "
              f"{row_stats.p95_ms:>8.2f}ms {row_stats.max_ms:>8.2f}ms")
    print(f"  {'TOTAL':<12} {budget.total_ms():>8.2f}ms")
    print(f"\n  a 31 Hz frame is 32.26 ms -> {budget.headroom_ms(31.0):+.2f} ms of headroom")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
