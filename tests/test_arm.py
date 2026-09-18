"""The hardware boundary, tested without hardware.

Every number quoted here was measured on our arm 2026-09-18 with the read-only
probe in `scripts/arm.py check`. Where a test asserts a specific figure it is
pinning a fact about OUR arm, and the docstring says so, because a different
arm calibrated differently will legitimately differ.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from robo_mimic.arm import (
    GRIPPER_UNITS,
    check_calibration,
    entry_pose,
    from_lerobot,
    report_pose,
    stale_goal_deg,
    to_lerobot,
)
from robo_mimic.kinematics.limits import JOINTS, LIMITS_DEG

# The pose our arm actually rests in, in lerobot degrees.
REST = {
    "shoulder_pan": -4.84,
    "shoulder_lift": -104.04,
    "elbow_flex": 96.13,
    "wrist_flex": 77.01,
    "wrist_roll": -71.78,
}

def _cal(id_: int, homing: int, lo: int, hi: int) -> dict[str, int]:
    return {"id": id_, "drive_mode": 0, "homing_offset": homing, "range_min": lo, "range_max": hi}


# phi's calibration for this follower, in ticks.
CALIBRATION = {
    "shoulder_pan": _cal(1, 1860, 791, 3415),
    "shoulder_lift": _cal(2, -1619, 805, 3180),
    "elbow_flex": _cal(3, 1291, 816, 3023),
    "wrist_flex": _cal(4, -1507, 940, 3282),
    "wrist_roll": _cal(5, 603, 0, 4095),
    "gripper": _cal(6, 462, 2032, 3544),
}


class TestUnitConversion:
    def test_body_joints_pass_through_as_degrees(self) -> None:
        action = to_lerobot(REST, 0.5)
        for name in JOINTS:
            assert action[f"{name}.pos"] == pytest.approx(REST[name])

    def test_gripper_rescales_openness_to_lerobot_units(self) -> None:
        assert to_lerobot(REST, 0.0)["gripper.pos"] == pytest.approx(0.0)
        assert to_lerobot(REST, 1.0)["gripper.pos"] == pytest.approx(GRIPPER_UNITS)
        assert to_lerobot(REST, 0.37)["gripper.pos"] == pytest.approx(37.0)

    @pytest.mark.parametrize("openness", [-5.0, -0.001, 1.001, 99.0])
    def test_openness_outside_zero_one_is_clamped_not_scaled(self, openness: float) -> None:
        """lerobot would clamp this silently. We would rather be the one who did."""
        value = to_lerobot(REST, openness)["gripper.pos"]
        assert 0.0 <= value <= GRIPPER_UNITS

    def test_round_trip_is_exact(self) -> None:
        joints, openness = from_lerobot(to_lerobot(REST, 0.42))
        assert joints == pytest.approx(REST)
        assert openness == pytest.approx(0.42)


class TestPoseReport:
    def test_our_arm_rests_outside_the_shoulder_lift_limit(self) -> None:
        """MEASURED on our arm. The rest pose is not inside the envelope we
        command, so power-up is not a no-op. This is the fact `entry_pose`
        exists to surface."""
        outside = [r for r in report_pose(REST) if not r.inside]
        assert [r.joint for r in outside] == ["shoulder_lift"]
        assert outside[0].excursion_deg == pytest.approx(4.04, abs=0.01)

    def test_excursion_is_zero_inside_the_limits(self) -> None:
        middle = {name: 0.0 for name in JOINTS}
        assert all(r.excursion_deg == 0.0 and r.inside for r in report_pose(middle))

    def test_report_covers_every_joint_we_command(self) -> None:
        assert [r.joint for r in report_pose(REST)] == list(JOINTS)


class TestEntryPose:
    def test_entry_pose_is_inside_the_limits_by_construction(self) -> None:
        clamped, _ = entry_pose(REST)
        for name, value in clamped.items():
            lo, hi = LIMITS_DEG[name]
            assert lo <= value <= hi

    def test_it_reports_the_uncommanded_move_it_is_about_to_cause(self) -> None:
        """The whole point: 4.04 deg of shoulder_lift that nobody asked for."""
        _, worst = entry_pose(REST)
        assert worst == pytest.approx(4.04, abs=0.01)

    def test_a_pose_already_inside_needs_no_correction(self) -> None:
        _, worst = entry_pose({name: 0.0 for name in JOINTS})
        assert worst == 0.0

    def test_only_the_offending_joint_moves(self) -> None:
        clamped, _ = entry_pose(REST)
        moved = [n for n in clamped if abs(clamped[n] - REST[n]) > 1e-9]
        assert moved == ["shoulder_lift"]


class TestCalibrationGate:
    def test_our_limits_are_a_subset_on_every_joint(self) -> None:
        """MEASURED. If this ever fails we are commanding into a hard stop."""
        for joint, calibrated, ours, subset in check_calibration(CALIBRATION):
            assert subset, f"{joint}: we command {ours:.2f} deg of a {calibrated:.2f} deg joint"

    def test_it_catches_a_joint_we_would_overdrive(self) -> None:
        narrow = {k: dict(v) for k, v in CALIBRATION.items()}
        narrow["shoulder_pan"]["range_max"] = narrow["shoulder_pan"]["range_min"] + 100
        failures = [row[0] for row in check_calibration(narrow) if not row[3]]
        assert failures == ["shoulder_pan"]

    def test_the_committed_calibration_matches_phis_file(self) -> None:
        """Guards against this test's copy drifting from the real calibration."""
        path = Path(
            "/Volumes/Crucial_X9/Projects/phi/configs/calibration"
            "/robots/so_follower/phi_follower.json"
        )
        if not path.exists():  # not everyone has phi checked out
            pytest.skip("phi calibration not present")
        assert json.loads(path.read_text()) == CALIBRATION


class TestStaleGoal:
    def test_our_arm_had_no_stale_target(self) -> None:
        """MEASURED 2026-09-18: 15 ticks worst case, so energising cannot snap."""
        present = {"shoulder_pan": 2048, "shoulder_lift": 809, "elbow_flex": 3013,
                   "wrist_flex": 2987, "wrist_roll": 1231, "gripper": 2047}
        goal = {"shoulder_pan": 2050, "shoulder_lift": 794, "elbow_flex": 3017,
                "wrist_flex": 2990, "wrist_roll": 1233, "gripper": 2043}
        assert stale_goal_deg(present, goal) == pytest.approx(1.32, abs=0.01)

    def test_a_stale_target_is_reported_in_degrees_not_ticks(self) -> None:
        """A whole turn of stale goal must read as 360 deg, not as 4095."""
        assert stale_goal_deg({"a": 0}, {"a": 4095}) == pytest.approx(360.0)
