#!/usr/bin/env python
"""Regenerate the committed landmark goldens. Run with `make fixtures`.

WHY THE FIXTURE STORES OUTPUT, NOT INPUT
----------------------------------------
The reference image is a third-party MediaPipe sample, so it is fetched by
`make assets` rather than committed. What IS committed is what we derived from
it: the landmarks, plus hashes that let a test prove it is looking at the same
pixels we were.

WHY PNG AND NOT THE JPEG WE DOWNLOADED
--------------------------------------
Measured 2026-09-16. Decoding the same progressive JPEG with MediaPipe's loader
and with OpenCV gives images differing by at most 3 levels on 2.79 percent of
pixels -- a mean difference of 0.0377 levels. That microscopic difference moves
world landmarks by 0.7370 mm.

    pixels identical, image format differs (SRGBA vs SRGB) -> 0.0000 mm
    image format identical, JPEG decoder differs           -> 0.7370 mm

So the decoder, not the format, is the hazard. Transcoding once to PNG removes
it: PNG is lossless, every decoder produces identical bytes, and the landmarks
then match to 0.000000 mm. The fixture is reproducible by construction rather
than by hoping everyone uses the same JPEG library.
"""

from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from robo_mimic.landmarks import (  # noqa: E402
    LANDMARK_NAMES,
    HandTracker,
    read_image_rgb,
)

MODEL = ROOT / "assets" / "hand_landmarker.task"
SOURCE_JPEG = ROOT / "fixtures" / "images" / "woman_hands.jpg"
REFERENCE_PNG = ROOT / "fixtures" / "images" / "reference_hands.png"
OUT = ROOT / "fixtures" / "landmarks_golden.npz"

MEDIAPIPE_PIN = "0.10.35"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    if not MODEL.exists() or not SOURCE_JPEG.exists():
        print("missing assets. Run `make assets` first.", file=sys.stderr)
        return 1

    import cv2
    import mediapipe

    if mediapipe.__version__ != MEDIAPIPE_PIN:
        print(
            f"refusing: fixtures must be generated with mediapipe {MEDIAPIPE_PIN}, "
            f"found {mediapipe.__version__}. The landmarks are version-specific.",
            file=sys.stderr,
        )
        return 1

    # Transcode once, losslessly, so every later decode agrees.
    if not REFERENCE_PNG.exists():
        cv2.imwrite(
            str(REFERENCE_PNG),
            cv2.imread(str(SOURCE_JPEG)),
            [cv2.IMWRITE_PNG_COMPRESSION, 9],
        )
        print(f"transcoded {SOURCE_JPEG.name} -> {REFERENCE_PNG.name}")

    rgb = read_image_rgb(REFERENCE_PNG)
    pixel_hash = sha256(rgb.tobytes())

    with HandTracker(MODEL, num_hands=2, mode="image") as tracker:
        tracker.detect(rgb)  # warm up; the first call is not representative
        timings = []
        for _ in range(100):
            start = time.perf_counter()
            frame = tracker.detect(rgb)
            timings.append((time.perf_counter() - start) * 1000.0)

    hands = sorted(frame.hands, key=lambda h: h.handedness)
    if len(hands) != 2:
        print(f"expected 2 hands in the reference image, got {len(hands)}", file=sys.stderr)
        return 1

    latency = np.array(timings)
    np.savez_compressed(
        OUT,
        image=np.stack([h.image for h in hands]),
        world=np.stack([h.world for h in hands]),
        handedness=np.array([h.handedness for h in hands]),
        score=np.array([h.score for h in hands]),
        landmark_names=np.array(LANDMARK_NAMES),
        mediapipe_version=np.array(mediapipe.__version__),
        source_jpeg_sha256=np.array(sha256(SOURCE_JPEG.read_bytes())),
        reference_png_sha256=np.array(sha256(REFERENCE_PNG.read_bytes())),
        decoded_pixels_sha256=np.array(pixel_hash),
        image_shape=np.array(rgb.shape),
        latency_ms_p50=np.array(float(np.percentile(latency, 50))),
        latency_ms_p90=np.array(float(np.percentile(latency, 90))),
        latency_ms_p99=np.array(float(np.percentile(latency, 99))),
    )

    print(f"\nwrote {OUT.relative_to(ROOT)}  ({OUT.stat().st_size / 1024:.1f} KB)")
    print(f"  mediapipe        {mediapipe.__version__}")
    print(f"  image            {rgb.shape}  decoded sha256 {pixel_hash[:16]}...")
    for hand in hands:
        print(
            f"  {hand.handedness:<5} score={hand.score:.4f}  "
            f"palm span={hand.palm_span_m * 100:5.2f} cm  pinch ratio={hand.pinch_ratio:.3f}"
        )
    print(
        f"  latency          p50 {np.percentile(latency, 50):.2f} ms  "
        f"p90 {np.percentile(latency, 90):.2f} ms  p99 {np.percentile(latency, 99):.2f} ms"
        f"  -> {1000 / np.percentile(latency, 90):.0f} FPS at p90"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
