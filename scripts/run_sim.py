#!/usr/bin/env python
"""Drive the sim through the whole pipeline and report tracking error.

    make sim                       the measurement table
    make sim ARGS="--video out.mp4"   plus a rendered clip

WHY A SYNTHETIC HAND AND NOT A RECORDED CLIP
--------------------------------------------
A recorded clip has no ground truth: you can see whether the arm followed, but
not by how much it should have. A synthetic trajectory is generated from the
committed fixture's REAL landmarks -- a real hand shape, real MediaPipe output --
translated and scaled along a KNOWN path, with optional noise at the measured
level. So every error below is measured against something exact.

`make replay CLIP=...` drives the identical pipeline from a real recording once
one exists. This script is for the numbers.

WHAT RUNS
    fixture landmarks -> moved on a known path -> HandLandmarks
      -> hand_pose / image_position -> Retargeter -> project -> SafetyLimiter
      -> SimArm (500 Hz, interpolating) -> measure
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from robo_mimic.landmarks import MIDDLE_MCP, PALM, WRIST, HandLandmarks  # noqa: E402
from robo_mimic.project import project  # noqa: E402
from robo_mimic.retarget import Clutch, RetargetConfig, Retargeter  # noqa: E402
from robo_mimic.safety import SafetyConfig, SafetyLimiter  # noqa: E402
from robo_mimic.sim import SimArm  # noqa: E402
from robo_mimic.types import Pose  # noqa: E402

GOLDEN = ROOT / "fixtures" / "landmarks_golden.npz"
SCENE = ROOT / "model" / "scene.xml"
PERCEPTION_HZ = 31.0

#: Where the arm starts. Mid-workspace, reachable, elbow comfortably bent.
START = {
    "shoulder_pan": 0.0,
    "shoulder_lift": -40.0,
    "elbow_flex": 55.0,
    "wrist_flex": -25.0,
    "wrist_roll": 0.0,
}


def synthetic_hand(
    base: HandLandmarks,
    offset_xy: tuple[float, float],
    span_scale: float,
    pinch_ratio: float,
    noise_units: float,
    rng: np.random.Generator,
) -> HandLandmarks:
    """The fixture's real hand, moved on screen and re-pinched.

    Image landmarks are scaled about the palm centre (which is what changes the
    apparent span, i.e. depth) then translated. World landmarks carry the pinch,
    set by distance so the ratio is exact.
    """
    image = base.image.copy()
    centre = image[list(PALM), :2].mean(axis=0)
    image[:, :2] = centre + (image[:, :2] - centre) * span_scale
    image[:, 0] += offset_xy[0]
    image[:, 1] += offset_xy[1]
    if noise_units:
        image[:, :2] += rng.normal(0.0, noise_units, image[:, :2].shape)

    world = base.world.copy()
    span = float(np.linalg.norm(world[WRIST] - world[MIDDLE_MCP]))
    world[4] = world[8] + np.array([pinch_ratio * span, 0.0, 0.0])
    return HandLandmarks(
        image=image, world=world, handedness=base.handedness, score=base.score,
        timestamp_ms=0,
    )


def trajectory(seconds: float, hz: float) -> list[tuple[float, float, float, float]]:
    """(dx, dy, span_scale, pinch) per frame. A reach, a grasp, a lift, a release."""
    frames = int(seconds * hz)
    out = []
    for index in range(frames):
        t = index / hz
        phase = t / seconds
        dx = 0.10 * np.sin(2 * np.pi * 0.35 * t)
        dy = -0.06 * np.sin(2 * np.pi * 0.25 * t)
        span = 1.0 + 0.18 * np.sin(2 * np.pi * 0.2 * t)
        # open, close around a third of the way in, hold, release near the end
        pinch = 0.9 if phase < 0.3 else (0.12 if phase < 0.75 else 0.9)
        out.append((dx, dy, span, pinch))
    return out


def run(
    *,
    interpolate: bool,
    noise_units: float,
    seconds: float,
    video: Path | None = None,
) -> dict[str, float]:
    with np.load(GOLDEN) as data:
        base = HandLandmarks(
            image=data["image"][0], world=data["world"][0],
            handedness=str(data["handedness"][0]), score=float(data["score"][0]),
            timestamp_ms=0,
        )

    rng = np.random.default_rng(20260917)
    retargeter = Retargeter(RetargetConfig(aspect=960 / 640))
    limiter = SafetyLimiter(SafetyConfig())
    limiter.reset(START)

    arm = SimArm(SCENE, interpolate=interpolate, start=START)
    dt = 1.0 / PERCEPTION_HZ

    renderer = frames = None
    if video is not None:
        renderer = arm.renderer()
        frames = []

    joint_err: list[float] = []
    peak_err: list[float] = []
    pos_err: list[float] = []
    engaged_frames = 0
    held_frames = 0

    for index, (dx, dy, span, pinch) in enumerate(trajectory(seconds, PERCEPTION_HZ)):
        hand = synthetic_hand(base, (dx, dy), span, pinch, noise_units, rng)
        timestamp = int(index * dt * 1000)

        tool_now = Pose(position=arm.pose().position, rotation=arm.pose().rotation)
        command = retargeter.step(hand, tool_now, timestamp, engage=True)
        if command.clutch is Clutch.ENGAGED:
            engaged_frames += 1

        joints = gripper = None
        if command.target is not None:
            joints = project(command.target, current=limiter.last).joints
            gripper = command.gripper

        safe = limiter.step(joints, gripper, dt)
        held_frames += safe.held
        arm.set_target(safe.joints, safe.gripper)
        step = arm.advance(dt)

        joint_err.append(step.joint_error_deg)
        peak_err.append(step.peak_joint_error_deg)
        pos_err.append(step.position_error_m)

        if renderer is not None and frames is not None and index % 2 == 0:
            frames.append(arm.frame(renderer))

    if video is not None and frames:
        _write_video(frames, video, PERCEPTION_HZ / 2)

    arm.close()
    return {
        "settled_p50": float(np.percentile(joint_err, 50)),
        "settled_p95": float(np.percentile(joint_err, 95)),
        "peak_p50": float(np.percentile(peak_err, 50)),
        "peak_p95": float(np.percentile(peak_err, 95)),
        "peak_max": float(np.max(peak_err)),
        "pos_p95_mm": float(np.percentile(pos_err, 95)) * 1000,
        "engaged": engaged_frames,
        "held": held_frames,
        "frames": len(joint_err),
    }


def _write_video(frames: list[np.ndarray], path: Path, fps: float) -> None:
    import cv2

    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    for frame in frames:
        writer.write(np.ascontiguousarray(frame[:, :, ::-1]))
    writer.release()
    print(f"\nwrote {path}  ({len(frames)} frames, {fps:.0f} fps)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--video", type=Path, help="render a clip to this path")
    parser.add_argument(
        "--noise",
        type=float,
        default=0.00035,
        help="image-landmark noise in normalized units (default = the measured "
        "level at sigma = 1 brightness level)",
    )
    args = parser.parse_args()

    if not SCENE.exists():
        print("model missing. Run `make model`.", file=sys.stderr)
        return 1
    if not GOLDEN.exists():
        print("fixture missing. Run `make fixtures`.", file=sys.stderr)
        return 1

    print(f"\nSO-101 in MuJoCo, {PERCEPTION_HZ:.0f} Hz perception / 500 Hz physics")
    print(f"synthetic hand from the committed fixture, {args.seconds:.0f} s, "
          f"noise {args.noise} nu\n")

    print("=== the two-loop split: does interpolating actually matter? ===")
    print(f"  {'fast loop':<28} {'settled p95':>12} {'PEAK p95':>10} {'peak max':>10}")
    results = {}
    for label, interpolate in (("hold (31 Hz staircase)", False),
                               ("interpolate (500 Hz)", True)):
        stats = run(interpolate=interpolate, noise_units=args.noise,
                    seconds=args.seconds)
        results[label] = stats
        print(f"  {label:<28} {stats['settled_p95']:>11.4f}d "
              f"{stats['peak_p95']:>9.4f}d {stats['peak_max']:>9.4f}d")
    a = results["hold (31 Hz staircase)"]["peak_p95"]
    b = results["interpolate (500 Hz)"]["peak_p95"]
    print(f"\n  interpolating cuts peak tracking error {a / max(b, 1e-9):.1f}x")

    print("\n=== noise sensitivity, interpolating ===")
    print(f"  {'image noise (nu)':>18} {'settled p95':>12} {'PEAK p95':>10} {'tool p95':>10}")
    for noise in (0.0, 0.00035, 0.001, 0.002):
        stats = run(interpolate=True, noise_units=noise, seconds=args.seconds)
        print(f"  {noise:>18.5f} {stats['settled_p95']:>11.4f}d "
              f"{stats['peak_p95']:>9.4f}d {stats['pos_p95_mm']:>8.3f}mm")

    final = run(interpolate=True, noise_units=args.noise, seconds=args.seconds,
                video=args.video)
    print(f"\n  {final['frames']} perception frames, {final['engaged']} engaged, "
          f"{final['held']} held")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
