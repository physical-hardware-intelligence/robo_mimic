"""Shared fixtures. Nothing here touches a camera, a robot, or the network."""

from __future__ import annotations

import os
import pathlib
from typing import Any

import numpy as np
import pytest

FIXTURES = pathlib.Path(__file__).resolve().parent.parent / "fixtures"

#: Where the phi repo lives, for the `phi`-marked provenance tests -- the ones
#: that import phi's own kinematics and demand bit-identical output from our
#: vendored copy. Set `PHI_REPO` to point at a checkout; without it those tests
#: skip rather than fail, because provenance is verifiable at home and is not a
#: precondition for the suite.
PHI_REPO = pathlib.Path(
    os.environ.get("PHI_REPO", "/Volumes/Crucial_X9/Projects/phi")
).expanduser()
PHI_SIM = PHI_REPO / "simulation"
PHI_MODEL = PHI_SIM / "model" / "so101_new_calib.xml"

#: Fetched by `make assets`, never committed (7.8 MB model + a third-party image).
MODEL = pathlib.Path(__file__).resolve().parent.parent / "assets" / "hand_landmarker.task"
REFERENCE_PNG = FIXTURES / "images" / "reference_hands.png"

#: Fetched by `make model` (16.4 MB of meshes), never committed.
SCENE = pathlib.Path(__file__).resolve().parent.parent / "model" / "scene.xml"


@pytest.fixture(scope="session")
def golden() -> dict[str, Any]:
    """400 joint configurations and their FK results, generated from the phi originals."""
    with np.load(FIXTURES / "kinematics_golden.npz") as data:
        return {k: data[k] for k in data.files}


@pytest.fixture(scope="session")
def landmarks_golden() -> dict[str, Any]:
    """Landmarks frozen from the reference image with mediapipe 0.10.35.

    The image itself is a third-party sample, fetched by `make assets` rather
    than committed. What is committed is what we derived from it, plus hashes so
    a test can prove it is looking at the same pixels.
    """
    with np.load(FIXTURES / "landmarks_golden.npz") as data:
        return {k: data[k] for k in data.files}


@pytest.fixture(scope="session")
def golden_hands(landmarks_golden: dict[str, Any]) -> list[Any]:
    """The golden fixture rehydrated into HandLandmarks objects.

    Imported here rather than at module scope so that collecting this file never
    requires mediapipe -- robo_mimic.landmarks imports it lazily, inside HandTracker.
    """
    from robo_mimic.landmarks import HandLandmarks

    g = landmarks_golden
    return [
        HandLandmarks(
            image=g["image"][i],
            world=g["world"][i],
            handedness=str(g["handedness"][i]),
            score=float(g["score"][i]),
            timestamp_ms=0,
        )
        for i in range(len(g["handedness"]))
    ]


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded, so a failure is reproducible from the test name alone."""
    return np.random.default_rng(20260916)
