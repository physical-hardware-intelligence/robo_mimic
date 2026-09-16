"""Shared fixtures. Nothing here touches a camera, a robot, or the network."""

from __future__ import annotations

import pathlib
from typing import Any

import numpy as np
import pytest

FIXTURES = pathlib.Path(__file__).resolve().parent.parent / "fixtures"

#: Where the phi repo lives, for the provenance tests. Absent on CI, and the
#: tests that need it skip rather than fail -- provenance is verifiable at home,
#: not a precondition for the suite.
PHI_MODEL = pathlib.Path(
    "/Volumes/Crucial_X9/Projects/phi/simulation/model/so101_new_calib.xml"
)
PHI_SIM = pathlib.Path("/Volumes/Crucial_X9/Projects/phi/simulation")


@pytest.fixture(scope="session")
def golden() -> dict[str, Any]:
    """400 joint configurations and their FK results, generated from the phi originals."""
    with np.load(FIXTURES / "kinematics_golden.npz") as data:
        return {k: data[k] for k in data.files}


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded, so a failure is reproducible from the test name alone."""
    return np.random.default_rng(20260916)
