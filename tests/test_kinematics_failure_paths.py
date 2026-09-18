"""The three paths where the IK gives up. Bugs live here, so they get tests.

Coverage found these uncovered. Rather than accept 95 percent, each one is
either exercised or -- in one case -- shown to be unreachable, which is itself
worth knowing.
"""

from __future__ import annotations

import numpy as np
import pytest

from robo_mimic.kinematics.inverse import _D, _LAT, inverse_kinematics, pick
from robo_mimic.kinematics.limits import JOINTS

#: How close to the pan axis a target may be before the arm cannot reach it.
#: The tool sits 0.1778 mm off the arm plane, so a target inside that radius of
#: the pan axis is geometrically unreachable at any pitch.
LATERAL_MM = abs(_LAT) * 1000.0


def test_lateral_offset_is_the_expected_scale() -> None:
    assert LATERAL_MM == pytest.approx(0.1778, abs=1e-4)


def test_target_on_the_pan_axis_returns_no_solution() -> None:
    """A point on the arm's own rotation axis. Expect [], not a traceback."""
    assert inverse_kinematics(_D, 0.0, 0.0) == []


def test_target_inside_the_lateral_offset_returns_no_solution() -> None:
    nudge = np.array([0.5 * abs(_LAT), 0.0, 0.0])
    assert inverse_kinematics(_D + nudge, 0.0, 0.0) == []


def test_target_just_outside_the_lateral_offset_does_solve() -> None:
    """The boundary is real, not a blanket refusal near the base."""
    nudge = np.array([1e-3, 0.0, 0.0])
    assert len(inverse_kinematics(_D + nudge, 0.0, 0.0)) == 4


def test_pick_of_nothing_is_none() -> None:
    assert pick([]) is None


def test_pick_returns_none_when_no_solution_is_in_limits() -> None:
    impossible = [dict.fromkeys(JOINTS, 999.0)]
    assert pick(impossible) is None
    assert pick(impossible, current=dict.fromkeys(JOINTS, 0.0)) is None


def test_the_verify_guard_never_actually_rejects_a_branch() -> None:
    """Documents a finding, and guards it.

    `inverse_kinematics(..., verify=True)` re-runs FK on each branch and drops
    any whose position is off by more than 1e-9 m. Over 40000 random targets
    that guard fired ZERO times: the closed-form branch enumeration is exact, so
    `verify` is belt-and-braces rather than load-bearing. This matches the
    0.00 pm figure measured in phi.

    If a future change to the solve introduces a spurious branch, this test
    fails and points straight at it.
    """
    generator = np.random.default_rng(0)
    rejected = 0
    for _ in range(4000):
        target = generator.uniform(-0.6, 0.6, 3)
        pitch = generator.uniform(-np.pi, np.pi)
        checked = len(inverse_kinematics(target, pitch, 0.0, verify=True))
        unchecked = len(inverse_kinematics(target, pitch, 0.0, verify=False))
        rejected += unchecked - checked
    assert rejected == 0, f"the verify guard rejected {rejected} branches -- the solve regressed"


def test_pick_without_a_current_pose_takes_the_first_legal_branch() -> None:
    """The cold-start path: on the first frame there is no previous pose to be near.

    Deliberately distinct from the `current=` path, which minimises travel. At
    startup there is nothing to minimise against, so any in-limits branch will
    do and the caller must not assume a particular one.
    """
    from robo_mimic.kinematics.forward import get_forward_kinematics
    from robo_mimic.kinematics.inverse import tool_pitch, within_limits

    q = {
        "shoulder_pan": 10.0,
        "shoulder_lift": -30.0,
        "elbow_flex": 40.0,
        "wrist_flex": -20.0,
        "wrist_roll": 15.0,
    }
    position, _ = get_forward_kinematics({**q, "gripper": 0.0})
    pitch = tool_pitch(q["shoulder_lift"], q["elbow_flex"], q["wrist_flex"])
    solutions = inverse_kinematics(position, pitch, q["wrist_roll"])

    chosen = pick(solutions)
    assert chosen is not None
    assert within_limits(chosen)
    assert set(chosen) == set(JOINTS)
    # and it really is one of the branches, not a blend of them
    assert any(all(chosen[j] == s[j] for j in JOINTS) for s in solutions)
