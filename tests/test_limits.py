"""Joint limits: derived from the model, and provably not wider than it.

This file is the reason `limits.py` exists. It encodes, as an executable test,
the bug it was written to avoid.
"""

from __future__ import annotations

import math
import re

import pytest

from mirror.kinematics import limits as L

from .conftest import PHI_MODEL

# The hand-typed table in phi/simulation/so101_inverse_kinematics.py.
# Kept here as a REGRESSION RECORD: three of its five rows are wider than the
# model permits, which is what motivated deriving ours instead.
PHI_HAND_TYPED = {
    "shoulder_pan": (-110.0, 110.0),
    "shoulder_lift": (-100.0, 100.0),
    "elbow_flex": (-96.8, 96.8),
    "wrist_flex": (-95.0, 95.0),
    "wrist_roll": (-157.2, 162.8),
}


def test_joint_order_is_the_kinematic_chain_order() -> None:
    assert L.JOINTS == (
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
    )


def test_gripper_is_not_an_arm_joint() -> None:
    """Measured in phi: the jaw moves the tool frame by exactly 0.0 m."""
    assert "gripper" not in L.JOINTS
    assert "gripper" not in L.LIMITS_DEG


def test_degrees_are_computed_not_typed() -> None:
    for joint, (lo, hi) in L.CTRLRANGE_RAD.items():
        assert L.LIMITS_DEG[joint] == (math.degrees(lo), math.degrees(hi))


@pytest.mark.parametrize("joint", list(PHI_HAND_TYPED))
def test_our_limits_never_exceed_the_model(joint: str) -> None:
    """The invariant that matters: we are never more permissive than the robot."""
    lo, hi = L.LIMITS_DEG[joint]
    model_lo = math.degrees(L.CTRLRANGE_RAD[joint][0])
    model_hi = math.degrees(L.CTRLRANGE_RAD[joint][1])
    assert lo >= model_lo
    assert hi <= model_hi


def test_phi_hand_typed_table_is_wider_than_the_model() -> None:
    """Documents the defect. If phi ever fixes it, this test tells us to drop this file."""
    over = [
        j
        for j, (lo, hi) in PHI_HAND_TYPED.items()
        if lo < L.LIMITS_DEG[j][0] - 1e-12 or hi > L.LIMITS_DEG[j][1] + 1e-12
    ]
    assert sorted(over) == ["shoulder_pan", "wrist_flex", "wrist_roll"]


@pytest.mark.parametrize("joint", list(PHI_HAND_TYPED))
def test_clamp_is_idempotent_and_in_range(joint: str) -> None:
    lo, hi = L.LIMITS_DEG[joint]
    for value in (-1e6, lo - 1.0, lo, 0.0, hi, hi + 1.0, 1e6):
        clamped = L.clamp_deg(joint, value)
        assert lo <= clamped <= hi
        assert L.clamp_deg(joint, clamped) == clamped


def test_within_limits_rejects_one_bad_joint() -> None:
    ok = dict.fromkeys(L.JOINTS, 0.0)
    assert L.within_limits_deg(ok)
    bad = {**ok, "wrist_roll": 200.0}
    assert not L.within_limits_deg(bad)
    assert L.within_limits_deg(bad, tol_deg=40.0)


def test_spans_are_positive() -> None:
    assert all(L.span_deg(j) > 0 for j in L.JOINTS)


@pytest.mark.phi
def test_limits_still_match_the_model_file_on_disk() -> None:
    """Provenance. Re-parses the XML so a model edit cannot slip past us."""
    if not PHI_MODEL.exists():
        pytest.skip(f"phi model not reachable at {PHI_MODEL}")
    text = PHI_MODEL.read_text()
    found = {
        name: (float(lo), float(hi))
        for name, lo, hi in re.findall(
            r'<position class="sts3215" name="(\w+)".*?ctrlrange="([-\d.]+) ([-\d.]+)"', text
        )
    }
    assert found, "ctrlrange regex matched nothing -- the model format changed"
    for joint, value in L.CTRLRANGE_RAD.items():
        assert found[joint] == value, f"{joint} drifted: model {found[joint]} vs ours {value}"
    assert found["gripper"] == L.GRIPPER_RAD
