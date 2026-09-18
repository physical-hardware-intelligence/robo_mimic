"""The MuJoCo sink and the two-loop split.

Marked `sim`: needs mujoco and the fetched model, so `make test` skips it.
"""

from __future__ import annotations

import numpy as np
import pytest

from mirror.kinematics import forward_kinematics
from mirror.kinematics.limits import JOINTS
from mirror.sim import MODEL_JOINTS, SimArm

from .conftest import SCENE

pytestmark = pytest.mark.sim

START = {
    "shoulder_pan": 0.0,
    "shoulder_lift": -40.0,
    "elbow_flex": 55.0,
    "wrist_flex": -25.0,
    "wrist_roll": 0.0,
}
DT = 1.0 / 31.0


@pytest.fixture(autouse=True)
def _require_model() -> None:
    if not SCENE.exists():
        pytest.skip(f"model not fetched -- run `make model` ({SCENE})")


def test_a_missing_model_says_what_to_do() -> None:
    with pytest.raises(FileNotFoundError, match="Run `make model`"):
        SimArm("model/does_not_exist.xml")


def test_the_model_has_the_joints_we_expect() -> None:
    with SimArm(SCENE) as arm:
        assert MODEL_JOINTS == (*JOINTS, "gripper")
        assert arm.timestep_s == pytest.approx(0.002)


def test_teleport_places_the_arm_exactly() -> None:
    with SimArm(SCENE, start=START) as arm:
        for joint, angle in START.items():
            assert arm.joints[joint] == pytest.approx(angle, abs=1e-9)


def test_the_simulated_pose_matches_our_own_forward_kinematics() -> None:
    """`pose()` derives the tool from `qpos` through the FK already verified
    against `mj_forward` to 8.98e-09 m, so this pins the plumbing, not the FK."""
    with SimArm(SCENE, start=START) as arm:
        position, rotation = forward_kinematics({**START, "gripper": 0.0})
        assert np.abs(arm.pose().position - position).max() < 1e-12
        assert np.abs(arm.pose().rotation - rotation).max() < 1e-12


def test_it_converges_on_a_held_target() -> None:
    """The arm must actually get there, not merely head that way."""
    target = {**START, "shoulder_pan": 20.0, "elbow_flex": 40.0}
    with SimArm(SCENE, start=START) as arm:
        arm.set_target(target, 0.0)
        for _ in range(60):
            step = arm.advance(DT)
        assert step.joint_error_deg < 0.2, f"settled {step.joint_error_deg:.4f} deg"


def test_the_jaw_is_kinematically_decoupled_but_not_dynamically() -> None:
    """Two different kinds of decoupling, and only one of them is exact.

    KINEMATICALLY the jaw is perfectly decoupled: `forward_kinematics` with the
    gripper at 0 and at 1.5 rad gives tool poses differing by exactly
    0.000e+00 m. Nothing about the arm's geometry depends on the jaw.

    DYNAMICALLY it is not. Opening the jaw moves real mass, which changes the
    gravity torque on the arm, which changes its steady-state sag. Measured
    here: 3.19e-06 m of tool movement between fully open and fully shut. Real
    physics, and small enough not to matter -- but not zero, and a test that
    demanded zero was asserting something false.
    """
    kinematic_open, _ = forward_kinematics({**START, "gripper": 1.5})
    kinematic_shut, _ = forward_kinematics({**START, "gripper": 0.0})
    assert np.abs(kinematic_open - kinematic_shut).max() == 0.0

    with SimArm(SCENE, start=START) as arm:
        arm.set_target(START, 1.0)
        for _ in range(80):
            arm.advance(DT)
        opened = arm.pose().position
        arm.set_target(START, 0.0)
        for _ in range(80):
            arm.advance(DT)
        shift = float(np.abs(arm.pose().position - opened).max())
    assert shift < 5e-5, f"jaw moved the tool {shift:.2e} m -- more than sag explains"


def test_interpolating_beats_the_staircase() -> None:
    """THE PHASE 5 GATE.

    Perception at 31 Hz into actuators at 500 Hz with zeta = 0.250 and a 17.9 Hz
    resonance. Holding the target between updates is a staircase and rings it;
    ramping toward it does not. Measured over an 8 s synthetic trajectory:

        hold (31 Hz staircase)   peak p95 2.7884 deg, max 3.6516 deg
        interpolate (500 Hz)     peak p95 0.2089 deg, max 0.3608 deg   13.3x

    Here, a single large step, which is the worst case for both.
    """
    target = {**START, "shoulder_pan": 25.0, "shoulder_lift": -20.0}
    peaks = {}
    for interpolate in (False, True):
        with SimArm(SCENE, interpolate=interpolate, start=START) as arm:
            worst = 0.0
            for index in range(40):
                blend = min(1.0, index / 20)
                arm.set_target(
                    {j: START[j] + blend * (target[j] - START[j]) for j in JOINTS}, 0.0
                )
                worst = max(worst, arm.advance(DT).peak_joint_error_deg)
            peaks[interpolate] = worst
    assert peaks[True] < peaks[False], f"{peaks}"
    assert peaks[False] / peaks[True] > 2.0, f"only {peaks[False] / peaks[True]:.1f}x"


def test_peak_error_is_not_hidden_by_settled_error() -> None:
    """A step's ripple lives between substeps. Reporting only the settled value
    at the end of each tick would make the staircase look fine."""
    with SimArm(SCENE, interpolate=False, start=START) as arm:
        arm.set_target({**START, "shoulder_pan": 30.0}, 0.0)
        step = arm.advance(DT)
        assert step.peak_joint_error_deg >= step.joint_error_deg


def test_close_is_idempotent() -> None:
    arm = SimArm(SCENE, start=START)
    arm.renderer()
    arm.close()
    arm.close()


def test_rendering_produces_a_real_image() -> None:
    with SimArm(SCENE, start=START) as arm:
        frame = arm.frame(arm.renderer())
        assert frame.ndim == 3 and frame.shape[2] == 3
        assert frame.dtype == np.uint8
        assert frame.std() > 5.0, "a uniform frame means nothing was drawn"


def test_sim_time_advances_with_the_physics_not_the_wall_clock() -> None:
    with SimArm(SCENE, start=START) as arm:
        arm.set_target(START, 0.0)
        for _ in range(10):
            arm.advance(DT)
        expected = 10 * round(DT / 0.002) * 0.002
        assert arm.time_s == pytest.approx(expected, abs=1e-9)
