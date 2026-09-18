"""The guarantee: for any input, the output is legal.

This is the module that protects a physical arm, so its contract is stated as an
invariant rather than a set of examples, and fuzzed.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from robo_mimic.kinematics.limits import GRIPPER_RAD, JOINTS, LIMITS_DEG
from robo_mimic.safety import (
    OneEuroFilter,
    SafetyConfig,
    SafetyLimiter,
    all_in_limits,
    gripper_command_rad,
    velocity_of,
)

DT = 1.0 / 31.0
CONFIG = SafetyConfig()

#: Ways a caller can hand this module rubbish.
GARBAGE: list[dict[str, float] | None] = [
    None,
    dict.fromkeys(JOINTS, float("nan")),
    dict.fromkeys(JOINTS, float("inf")),
    dict.fromkeys(JOINTS, float("-inf")),
    dict.fromkeys(JOINTS, 1e9),
    dict.fromkeys(JOINTS, -1e9),
    {"shoulder_pan": 0.0},  # malformed: missing joints
    {},
]


def zeros() -> dict[str, float]:
    return dict.fromkeys(JOINTS, 0.0)


# --- the hard guarantee ---------------------------------------------------------
@pytest.mark.parametrize("bad", GARBAGE, ids=lambda b: str(type(b).__name__) + str(b)[:24])
def test_no_single_bad_input_escapes(bad: dict[str, float] | None) -> None:
    limiter = SafetyLimiter(CONFIG)
    start = dict(limiter.last)
    command = limiter.step(bad, 0.5, DT)

    assert all_in_limits(command.joints)
    assert 0.0 <= command.gripper <= 1.0
    step = max(abs(command.joints[j] - start[j]) for j in JOINTS)
    assert step <= CONFIG.max_velocity_deg_s * DT * (1 + 1e-9)

    # Unrepresentable input (None, NaN, inf, malformed) must be HELD. A merely
    # absurd but FINITE target -- 1e9 -- is a legitimate request, so it is
    # followed: rate-limited toward the target and clamped at the limit. That
    # distinction matters, because holding on any large value would freeze the
    # arm whenever the operator moved quickly.
    finite_request = isinstance(bad, dict) and set(bad) == set(JOINTS) and all(
        np.isfinite(v) for v in bad.values()
    )
    assert command.held is not finite_request


@pytest.mark.parametrize("bad_dt", [0.0, -1.0, float("nan"), float("inf")])
def test_a_bad_timestep_is_refused(bad_dt: float) -> None:
    """`dt` is used as a physical quantity; a bad one would mis-scale every cap."""
    limiter = SafetyLimiter(CONFIG)
    command = limiter.step(zeros(), 0.5, bad_dt)
    assert command.held
    assert command.reason == "bad dt"


def test_thirty_thousand_adversarial_frames_never_escape() -> None:
    """The Phase 4 gate.

    Mixed stream of None, NaN, +/-inf, 1e9, malformed dicts and wild finite
    values, with occasional bad timesteps. Measured: 0 limit escapes, 0 illegal
    jaw values, worst joint velocity exactly 120.0000 deg/s and worst jaw rate
    exactly 2.5000 /s -- AT the caps, never over.
    """
    rng = np.random.default_rng(3)
    limiter = SafetyLimiter(CONFIG)
    previous = dict(limiter.last)
    worst_velocity = worst_jaw = 0.0
    moved = 0

    for _ in range(30_000):
        roll = rng.random()
        if roll < 0.44:
            joints = GARBAGE[int(roll / 0.44 * len(GARBAGE)) % len(GARBAGE)]
        else:
            joints = {j: float(rng.uniform(-400, 400)) for j in JOINTS}
        gripper = None if rng.random() < 0.1 else float(rng.uniform(-5, 5))
        dt = DT if rng.random() < 0.9 else float(rng.choice([0.0, -1.0, np.nan, 1e-9, 1e6]))

        before_jaw = limiter.last_gripper
        command = limiter.step(joints, gripper, dt)

        assert all_in_limits(command.joints)
        assert np.isfinite(command.gripper) and 0.0 <= command.gripper <= 1.0
        if np.isfinite(dt) and dt > 0:
            speeds = velocity_of(command.joints, previous, dt)
            worst_velocity = max(worst_velocity, max(abs(v) for v in speeds.values()))
            worst_jaw = max(worst_jaw, abs(command.gripper - before_jaw) / dt)
        if max(abs(command.joints[j] - previous[j]) for j in JOINTS) > 1e-9:
            moved += 1
        previous = dict(command.joints)

    assert worst_velocity <= CONFIG.max_velocity_deg_s * (1 + 1e-9)
    assert worst_jaw <= CONFIG.max_gripper_rate_per_s * (1 + 1e-9)
    assert moved > 100, "a limiter that never moves is not a limiter, it is a brake"


@given(
    values=st.lists(st.floats(-500.0, 500.0, allow_nan=False), min_size=1, max_size=200),
    jaw=st.floats(-3.0, 3.0, allow_nan=False),
)
@settings(max_examples=200, deadline=None)
def test_any_finite_sequence_stays_legal(values: list[float], jaw: float) -> None:
    limiter = SafetyLimiter(CONFIG)
    previous = dict(limiter.last)
    for value in values:
        command = limiter.step(dict.fromkeys(JOINTS, value), jaw, DT)
        assert all_in_limits(command.joints)
        speeds = velocity_of(command.joints, previous, DT)
        assert max(abs(v) for v in speeds.values()) <= CONFIG.max_velocity_deg_s * (
            1 + 1e-9
        )
        previous = dict(command.joints)


# --- the two regressions --------------------------------------------------------
def test_there_is_no_first_command_exemption() -> None:
    """Regression: fuzzing found a 3410 deg/s escape in three frames.

    An earlier version adopted the FIRST command directly, reasoning that
    filtering from an arbitrary origin would drag the arm in from nowhere. But a
    couple of dropped frames report the home posture while leaving the internal
    state unset, so the next real command took the exemption and moved a joint
    109.9999 deg in a single frame.
    """
    limiter = SafetyLimiter(CONFIG)
    limiter.step(None, None, DT)  # dropped frame first
    limiter.step(None, None, DT)
    start = dict(limiter.last)
    command = limiter.step(dict.fromkeys(JOINTS, 90.0), 1.0, DT)
    step = max(abs(command.joints[j] - start[j]) for j in JOINTS)
    assert step <= CONFIG.max_velocity_deg_s * DT * (1 + 1e-9), f"{step:.4f} deg jump"


def test_the_glitch_guard_does_not_freeze_a_sustained_move() -> None:
    """Regression: guarding against the previous OUTPUT froze the arm forever.

    Because the output lags behind (the rate limiter working as intended), a
    sustained fast motion looked like a permanent discontinuity. Measured with
    that version: 30000 adversarial frames, worst joint velocity 0.0000 deg/s --
    nothing ever moved. The guard now compares input to PREVIOUS INPUT.
    """
    limiter = SafetyLimiter(CONFIG)
    target = 80.0
    for _ in range(200):
        command = limiter.step(dict.fromkeys(JOINTS, target), 0.5, DT)
    reached = command.joints["shoulder_pan"]
    assert reached == pytest.approx(target, abs=1.0), f"stalled at {reached:.2f}"


def test_a_one_frame_glitch_is_dropped() -> None:
    limiter = SafetyLimiter(CONFIG)
    for _ in range(40):
        limiter.step(dict.fromkeys(JOINTS, 10.0), 0.5, DT)
    settled = dict(limiter.last)
    glitch = limiter.step(dict.fromkeys(JOINTS, 200.0), 0.5, DT)
    assert glitch.held
    assert "glitch" in glitch.reason
    assert glitch.joints == pytest.approx(settled)  # type: ignore[arg-type]


# --- clamping, rate limiting, holding ------------------------------------------
def test_a_wild_target_is_clamped_to_the_model_not_the_vendored_table() -> None:
    """`limits.py` is the single source of truth. phi's hand-typed table is
    wider on three joints, and `wrist_roll` disagrees in both directions."""
    limiter = SafetyLimiter(CONFIG)
    for _ in range(4000):
        command = limiter.step(dict.fromkeys(JOINTS, 10_000.0), 1.0, DT)
    for joint in JOINTS:
        assert command.joints[joint] == pytest.approx(LIMITS_DEG[joint][1], abs=1e-9)
    assert joint in command.clamped or command.clamped, "clamping must be reported"


def test_rate_limiting_is_reported_and_respected() -> None:
    limiter = SafetyLimiter(CONFIG)
    command = limiter.step(dict.fromkeys(JOINTS, 50.0), 0.5, DT)
    assert set(command.rate_limited) >= set(JOINTS)
    assert max(abs(command.joints[j]) for j in JOINTS) <= CONFIG.max_velocity_deg_s * DT


def test_the_jaw_is_slower_than_the_arm_on_purpose() -> None:
    """A jaw slamming shut is the one motion that can crush something."""
    assert CONFIG.max_gripper_rate_per_s * DT < 0.1, "under 10% of travel per frame"
    limiter = SafetyLimiter(CONFIG)
    command = limiter.step(zeros(), 1.0, DT)
    assert command.gripper <= CONFIG.max_gripper_rate_per_s * DT + 1e-9


def test_holding_repeats_rather_than_retreating_home() -> None:
    """Commanding home on a dropped frame would make every glitch a full-speed
    retreat across the workspace."""
    limiter = SafetyLimiter(CONFIG)
    for _ in range(60):
        limiter.step(dict.fromkeys(JOINTS, 20.0), 0.5, DT)
    away = dict(limiter.last)
    assert abs(away["shoulder_pan"]) > 5.0, "should have travelled away from home"
    held = limiter.step(None, None, DT)
    assert held.held
    assert held.joints == pytest.approx(away)  # type: ignore[arg-type]


def test_reset_seeds_from_the_arms_actual_pose() -> None:
    """On hardware this is `Present_Position`, so the next command is measured
    from where the arm really is rather than from an assumed home."""
    limiter = SafetyLimiter(CONFIG)
    actual = {j: 25.0 for j in JOINTS}
    limiter.reset(actual, gripper=0.75)
    assert limiter.last == pytest.approx(actual)  # type: ignore[arg-type]
    command = limiter.step(dict.fromkeys(JOINTS, 25.0), 0.75, DT)
    assert max(abs(command.joints[j] - 25.0) for j in JOINTS) < 1e-6


def test_reset_clamps_an_out_of_range_actual_pose() -> None:
    limiter = SafetyLimiter(CONFIG)
    limiter.reset(dict.fromkeys(JOINTS, 999.0))
    assert all_in_limits(limiter.last)


# --- the filter ------------------------------------------------------------------
def test_config_rejects_nonsense() -> None:
    for kwargs in (
        {"max_velocity_deg_s": 0.0},
        {"max_gripper_rate_per_s": -1.0},
        {"min_cutoff_hz": 0.0},
        {"beta": -0.1},
        {"max_jump_deg": 0.0},
    ):
        with pytest.raises(ValueError):
            SafetyConfig(**kwargs)  # type: ignore[arg-type]


def test_the_filter_attenuates_a_stationary_signal() -> None:
    """Measured when tuned: 1.52 deg RMS in -> 0.69 deg out, a 2.2x cut."""
    rng = np.random.default_rng(0)
    noise = rng.normal(0.0, 1.5, 900)
    filt = OneEuroFilter(CONFIG.min_cutoff_hz, CONFIG.beta)
    out = np.array([filt(float(v), DT) for v in noise])
    assert out.std() < noise.std() / 1.8


def test_the_filter_beats_the_raw_signal_while_MOVING_too() -> None:
    """The criterion the defaults were tuned against, and the reason beta is
    0.10 rather than 0.02: below that, lag error while moving exceeds the noise
    removed, so the filter is a net loss exactly when the operator is acting.
    """
    rng = np.random.default_rng(2)
    steps = 1200
    t = np.arange(steps) * DT
    truth = np.clip((t - 1.0) * 60.0, 0.0, 25.0) + np.clip((t - 3.0) * 15.0, 0.0, 18.0)
    signal = truth + rng.normal(0.0, 1.5, steps)
    speed = np.abs(np.gradient(truth, DT))
    moving = speed > 5.0

    filt = OneEuroFilter(CONFIG.min_cutoff_hz, CONFIG.beta)
    out = np.array([filt(float(v), DT) for v in signal])
    filtered_error = float(np.sqrt(np.mean((out[moving] - truth[moving]) ** 2)))
    raw_error = float(np.sqrt(np.mean((signal[moving] - truth[moving]) ** 2)))
    assert filtered_error < raw_error, f"filtered {filtered_error:.3f} vs raw {raw_error:.3f}"


def test_the_cutoff_actually_adapts() -> None:
    filt = OneEuroFilter(CONFIG.min_cutoff_hz, CONFIG.beta)
    for _ in range(50):
        filt(0.0, DT)
    still = filt.cutoff_at(DT)
    for k in range(50):
        filt(float(k) * 3.0, DT)
    moving = filt.cutoff_at(DT)
    assert still == pytest.approx(CONFIG.min_cutoff_hz, abs=1e-6)
    assert moving > still * 2.0, f"still {still:.2f} Hz, moving {moving:.2f} Hz"


@given(value=st.floats(-1e12, 1e12, allow_nan=False), dt=st.floats(1e-4, 1.0))
@settings(max_examples=200, deadline=None)
def test_the_applied_cutoff_never_exceeds_nyquist(value: float, dt: float) -> None:
    """Regression: a 1e9 input drove the uncapped cutoff to 5.2e8 Hz, a filter
    fully open to a rate that cannot represent anything near it."""
    filt = OneEuroFilter(CONFIG.min_cutoff_hz, CONFIG.beta)
    for _ in range(6):
        filt(value, dt)
        assert filt.cutoff_at(dt) <= 0.5 / dt * (1 + 1e-9)


def test_the_stationary_cutoff_is_below_the_perception_nyquist() -> None:
    """Perception runs at ~31 Hz, so its Nyquist is 15.5 Hz -- and the arm's
    17.9 Hz resonance sits ABOVE that, unobservable. The filter must not trust
    anything near the fold."""
    assert CONFIG.min_cutoff_hz < 15.5 / 4


@given(
    value=st.floats(-1e6, 1e6, allow_nan=False),
    dt=st.floats(1e-6, 1.0, allow_nan=False),
)
@settings(max_examples=200, deadline=None)
def test_the_filter_never_produces_nonsense(value: float, dt: float) -> None:
    filt = OneEuroFilter(CONFIG.min_cutoff_hz, CONFIG.beta)
    for _ in range(5):
        out = filt(value, dt)
        assert np.isfinite(out)
    assert filt.cutoff_hz >= CONFIG.min_cutoff_hz


def test_filter_rejects_bad_parameters() -> None:
    with pytest.raises(ValueError):
        OneEuroFilter(0.0, 0.1)
    with pytest.raises(ValueError):
        OneEuroFilter(1.0, -0.1)


# --- actuator units --------------------------------------------------------------
@given(openness=st.floats(-10.0, 10.0, allow_nan=False))
@settings(max_examples=200, deadline=None)
def test_the_jaw_command_never_leaves_the_servo_range(openness: float) -> None:
    value = gripper_command_rad(openness)
    assert GRIPPER_RAD[0] <= value <= GRIPPER_RAD[1]
    assert np.isfinite(value)


# --- realistic operation ---------------------------------------------------------
def test_it_tracks_smooth_motion_and_holds_only_on_real_dropouts() -> None:
    """A limiter that is safe but unusable is not safe, it is broken.

    Measured: 2917 of 3000 frames moved, 83 held, against a 3 percent injected
    dropout rate.
    """
    rng = np.random.default_rng(7)
    limiter = SafetyLimiter(CONFIG)
    previous = dict(limiter.last)
    moved = held = 0
    worst = 0.0

    for index in range(3000):
        t = index * DT
        target = {
            joint: 30.0 * np.sin(2 * np.pi * 0.25 * t + k) + rng.normal(0.0, 1.5)
            for k, joint in enumerate(JOINTS)
        }
        command = limiter.step(None if rng.random() < 0.03 else target, 0.5, DT)
        assert all_in_limits(command.joints)
        worst = max(worst, max(abs(v) for v in velocity_of(command.joints, previous, DT).values()))
        held += command.held
        if max(abs(command.joints[j] - previous[j]) for j in JOINTS) > 1e-9:
            moved += 1
        previous = dict(command.joints)

    assert worst <= CONFIG.max_velocity_deg_s * (1 + 1e-9)
    assert moved > 2500, f"only moved {moved}/3000"
    assert 40 < held < 200, f"held {held}/3000, expected around 90"
