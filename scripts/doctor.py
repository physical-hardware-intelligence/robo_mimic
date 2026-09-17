#!/usr/bin/env python
"""Check everything needed to run the live viewer. `make doctor`.

Each line is PASS, FAIL with the exact fix, or SKIP. Exit code is the number of
failures, so it is usable in a script.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

GREEN, RED, YELLOW, DIM, OFF = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"
PIN = "0.10.35"

failures = 0


def check(label: str, ok: bool | None, detail: str = "", fix: str = "") -> None:
    global failures
    if ok is None:
        print(f"  {YELLOW}SKIP{OFF}  {label:<34} {DIM}{detail}{OFF}")
        return
    if ok:
        print(f"  {GREEN}PASS{OFF}  {label:<34} {DIM}{detail}{OFF}")
        return
    failures += 1
    print(f"  {RED}FAIL{OFF}  {label:<34} {detail}")
    if fix:
        for line in fix.splitlines():
            print(f"        {DIM}{line}{OFF}")


def main() -> int:
    print("\nmirror doctor\n")

    check("python", sys.version_info >= (3, 12), f"{sys.version.split()[0]}", "need 3.12+")

    try:
        import numpy

        check("numpy", True, numpy.__version__)
    except ImportError:
        check("numpy", False, "missing", "make setup")

    try:
        import mediapipe

        version = mediapipe.__version__
        check(
            "mediapipe",
            version == PIN,
            version,
            f"pinned to {PIN}. 1.0.1 aborts on macOS arm64 inside\n"
            f"TensorsToDetectionsCalculator. Fix:\n"
            f"  uv pip install --python $VIRTUAL_ENV/bin/python mediapipe=={PIN}",
        )
    except ImportError:
        check("mediapipe", False, "missing", 'make setup   (installs the [perception] extra)')

    try:
        import cv2

        check("opencv", True, cv2.__version__)
    except ImportError:
        cv2 = None  # type: ignore[assignment]
        check("opencv", False, "missing", "make setup")

    model = ROOT / "assets" / "hand_landmarker.task"
    check(
        "hand landmarker model",
        model.exists(),
        f"{model.stat().st_size / 1e6:.1f} MB" if model.exists() else "not fetched",
        "make assets",
    )

    golden = ROOT / "fixtures" / "landmarks_golden.npz"
    check("landmark fixture", golden.exists(), "committed" if golden.exists() else "missing",
          "make fixtures")

    # Camera. The permission belongs to whichever app launched this process, so
    # running from your own Terminal and running from an IDE are separate grants.
    if cv2 is None:
        check("camera", None, "opencv missing")
    else:
        camera = cv2.VideoCapture(0)
        opened = camera.isOpened()
        ok, frame = camera.read() if opened else (False, None)
        camera.release()
        if ok and frame is not None:
            height, width = frame.shape[:2]
            check("camera 0", True, f"{width}x{height}, aspect {height / width:.3f}")
        else:
            check(
                "camera 0",
                False,
                "not authorized",
                "macOS asks per-application, and the grant belongs to whichever\n"
                "app launched this process. Easiest path:\n"
                "  1. open Terminal.app\n"
                "  2. cd " + str(ROOT) + "\n"
                "  3. make view        <- click Allow when macOS asks\n"
                "If no prompt appears:\n"
                "  System Settings > Privacy & Security > Camera > enable Terminal",
            )

    # The whole pipeline, end to end, on the committed fixture. No camera needed.
    try:
        import numpy as np

        from mirror.handframe import hand_pose
        from mirror.landmarks import HandLandmarks
        from mirror.retarget import Clutch, Retargeter
        from mirror.types import Pose

        data = np.load(golden)
        hand = HandLandmarks(
            image=data["image"][0],
            world=data["world"][0],
            handedness=str(data["handedness"][0]),
            score=float(data["score"][0]),
            timestamp_ms=0,
        )
        rotation = hand_pose(hand).rotation
        orthonormal = bool(np.abs(rotation.T @ rotation - np.eye(3)).max() < 1e-9)

        retargeter = Retargeter()
        home = Pose(position=np.array([0.2, 0.0, 0.15]), rotation=np.eye(3))
        squeezed = HandLandmarks(
            image=hand.image,
            world=hand.world.copy(),
            handedness=hand.handedness,
            score=hand.score,
            timestamp_ms=0,
        )
        squeezed.world[4] = squeezed.world[8]  # thumb tip onto index tip -> pinch = 0
        engaged = retargeter.step(squeezed, home, 0).clutch is Clutch.ENGAGED
        check("pipeline (fixture -> command)", orthonormal and engaged,
              "frame orthonormal, clutch engages")
    except Exception as error:  # noqa: BLE001
        check("pipeline (fixture -> command)", False, f"{type(error).__name__}: {error}",
              "make check")

    print()
    if failures:
        print(f"  {RED}{failures} problem(s){OFF}. Fix the FAIL lines above, then rerun.\n")
    else:
        print(f"  {GREEN}all good{OFF}. Run: make view\n")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
