"""The clutch and the jaw: every state transition, plus the gripper mapping.

This module decides whether the arm moves and how hard the gripper squeezes.
Its whole behaviour is a table, so the table is the test.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mirror.kinematics import GRIPPER_RAD, gripper_openness, gripper_rad
from mirror.retarget import Clutch, RetargetConfig, Retargeter
from mirror.types import Pose

from .conftest_hands import make_hand, rodrigues

HOME = Pose(position=np.array([0.20, 0.0, 0.15]), rotation=np.eye(3))
CONFIG = RetargetConfig(
    scale_m_per_span=0.10, depth_scale_m_per_span=0.035, lost_grace_ms=200
)

MID = 0.475  # midway between pinch_closed and pinch_open -> jaw half open
WIDE = 0.95  # above pinch_open -> jaw fully open


def hand(ratio: float = WIDE, **kwargs: object) -> object:
    return make_hand(pinch_ratio=ratio, **kwargs)  # type: ignore[arg-type]


# --- config validation --------------------------------------------------------
@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"scale_m_per_span": 0.0}, "scales must be positive"),
        ({"depth_scale_m_per_span": -0.1}, "scales must be positive"),
        ({"pinch_closed": 0.9, "pinch_open": 0.4}, "pinch_closed < pinch_open"),
        ({"pinch_closed": -0.1}, "pinch_closed < pinch_open"),
        ({"lost_grace_ms": -1}, "grace must be non-negative"),
    ],
)
def test_bad_config_is_rejected(kwargs: dict[str, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        RetargetConfig(**kwargs)  # type: ignore[arg-type]


def test_orientation_is_off_by_default() -> None:
    """ADR-003: only 7-37% of pitches are reachable at a given position, so hand
    tilt setting the pitch would leave most frames with no IK solution."""
    assert RetargetConfig().follow_orientation is False


def test_depth_gain_is_smaller_than_lateral_by_default() -> None:
    """Depth carries 7-25x the noise for the same information."""
    config = RetargetConfig()
    assert config.depth_scale_m_per_span < config.scale_m_per_span
    assert np.array_equal(
        config.axis_gains,
        [config.scale_m_per_span, config.scale_m_per_span, config.depth_scale_m_per_span],
    )


# --- THE STATE TABLE ----------------------------------------------------------
# (start, hand present, engage held, ms since seen) -> (end state, commands?)
TABLE = [
    ("out,  hand, key up", Clutch.DISENGAGED, True, False, 0, Clutch.DISENGAGED, False),
    ("out,  hand, KEY DOWN", Clutch.DISENGAGED, True, True, 0, Clutch.ENGAGED, True),
    ("out,  no hand, key up", Clutch.DISENGAGED, False, False, 0, Clutch.DISENGAGED, False),
    ("out,  no hand, KEY DOWN", Clutch.DISENGAGED, False, True, 0, Clutch.DISENGAGED, False),
    ("in,   hand, key held", Clutch.ENGAGED, True, True, 0, Clutch.ENGAGED, True),
    ("in,   hand, KEY UP", Clutch.ENGAGED, True, False, 0, Clutch.DISENGAGED, False),
    ("in,   dropout 100 ms", Clutch.ENGAGED, False, True, 100, Clutch.ENGAGED, True),
    ("in,   dropout 200 ms", Clutch.ENGAGED, False, True, 200, Clutch.ENGAGED, True),
    ("in,   dropout 201 ms", Clutch.ENGAGED, False, True, 201, Clutch.DISENGAGED, False),
    ("in,   no hand, KEY UP", Clutch.ENGAGED, False, False, 0, Clutch.DISENGAGED, False),
]


@pytest.mark.parametrize(
    ("name", "start", "has_hand", "engage", "elapsed", "expect_state", "expect_command"),
    TABLE,
    ids=[row[0] for row in TABLE],
)
def test_every_clutch_transition(
    name: str,
    start: Clutch,
    has_hand: bool,
    engage: bool,
    elapsed: int,
    expect_state: Clutch,
    expect_command: bool,
) -> None:
    retargeter = Retargeter(CONFIG)
    time_ms = 1000

    if start is Clutch.ENGAGED:
        retargeter.step(hand(), HOME, time_ms, engage=True)  # type: ignore[arg-type]
        assert retargeter.clutch is Clutch.ENGAGED

    command = retargeter.step(  # type: ignore[arg-type]
        hand() if has_hand else None, HOME, time_ms + elapsed, engage=engage
    )

    assert command.clutch is expect_state, name
    assert (command.target is not None) is expect_command, name
    assert retargeter.clutch is expect_state, name


def test_the_jaw_is_gated_by_the_deadman_too() -> None:
    """A deadman that still lets the gripper move is not a deadman.

    `target` and `gripper` must appear and disappear together, in every row of
    the table. Otherwise releasing the key would still let the jaw crush.
    """
    for name, start, present, engage, elapsed, _state, expect in TABLE:
        retargeter = Retargeter(CONFIG)
        if start is Clutch.ENGAGED:
            retargeter.step(hand(), HOME, 1000, engage=True)  # type: ignore[arg-type]
        command = retargeter.step(  # type: ignore[arg-type]
            hand() if present else None, HOME, 1000 + elapsed, engage=engage
        )
        assert (command.gripper is not None) is expect, name
        assert (command.target is None) == (command.gripper is None), name


def test_a_key_is_binary_so_there_is_no_chatter() -> None:
    """The hysteresis band is gone because a key cannot hover at a threshold.

    Previously the pinch ratio needed two thresholds and a band to stop the
    clutch toggling every frame. A boolean removes the failure mode entirely.
    """
    retargeter = Retargeter(CONFIG)
    states = [
        retargeter.step(hand(ratio=0.1 + 0.85 * (i % 2)), HOME, i, engage=True).clutch  # type: ignore[arg-type]
        for i in range(20)
    ]
    assert set(states) == {Clutch.ENGAGED}, "wild pinch changes must not affect the clutch"


# --- the gripper --------------------------------------------------------------
@pytest.mark.parametrize(
    ("ratio", "expected"),
    [
        (0.00, 0.0),  # thumb on index
        (0.15, 0.0),  # exactly at pinch_closed
        (0.10, 0.0),  # below it -> clamped
        (0.475, 0.5),  # midway
        (0.80, 1.0),  # exactly at pinch_open
        (1.50, 1.0),  # beyond -> clamped
    ],
)
def test_pinch_ratio_maps_linearly_to_jaw_opening(ratio: float, expected: float) -> None:
    retargeter = Retargeter(CONFIG)
    command = retargeter.step(hand(ratio=ratio), HOME, 0, engage=True)  # type: ignore[arg-type]
    assert command.gripper == pytest.approx(expected, abs=1e-12)


def test_jaw_tracks_the_hand_continuously_not_as_a_switch() -> None:
    """The students' demo showed State 0.09 / 0.68 / 1.00. Proportional, not binary."""
    retargeter = Retargeter(CONFIG)
    openings = []
    for ratio in np.linspace(0.15, 0.80, 14):
        command = retargeter.step(hand(ratio=float(ratio)), HOME, 0, engage=True)  # type: ignore[arg-type]
        assert command.gripper is not None
        openings.append(command.gripper)
    assert np.all(np.diff(openings) > 0), "must be monotonic"
    assert len(set(np.round(openings, 6))) == len(openings), "must be continuous, not stepped"


@given(scale=st.floats(0.5, 2.0), ratio=st.floats(0.0, 1.5))
@settings(max_examples=200, deadline=None)
def test_jaw_opening_is_independent_of_hand_size(scale: float, ratio: float) -> None:
    """`pinch_ratio` divides by palm span, so a big hand and a small hand at the
    same relative pinch command the same jaw angle. No per-operator calibration."""
    small = Retargeter(CONFIG).step(hand(ratio=ratio), HOME, 0, engage=True)  # type: ignore[arg-type]
    big_hand = make_hand(pinch_ratio=ratio)
    scaled = type(big_hand)(
        image=big_hand.image,
        world=big_hand.world * scale,
        handedness=big_hand.handedness,
        score=big_hand.score,
        timestamp_ms=0,
    )
    big = Retargeter(CONFIG).step(scaled, HOME, 0, engage=True)
    assert small.gripper == pytest.approx(big.gripper, rel=1e-9)


# --- actuator units -----------------------------------------------------------
def test_gripper_rad_spans_the_model_ctrlrange() -> None:
    assert gripper_rad(0.0) == GRIPPER_RAD[0]
    assert gripper_rad(1.0) == GRIPPER_RAD[1]
    assert np.degrees(GRIPPER_RAD) == pytest.approx([-10.0, 100.0], abs=0.01)


@given(openness=st.floats(0.0, 1.0))
@settings(max_examples=200, deadline=None)
def test_gripper_units_round_trip(openness: float) -> None:
    assert gripper_openness(gripper_rad(openness)) == pytest.approx(openness, abs=1e-12)


@pytest.mark.parametrize("out_of_range", [-5.0, -0.001, 1.001, 99.0])
def test_gripper_rad_clamps_rather_than_exceeding_the_servo(out_of_range: float) -> None:
    """The jaw must never be commanded outside the model's own ctrlrange."""
    assert GRIPPER_RAD[0] <= gripper_rad(out_of_range) <= GRIPPER_RAD[1]


# --- what it commands ---------------------------------------------------------
def test_engaging_commands_exactly_where_the_arm_already_is() -> None:
    retargeter = Retargeter(CONFIG)
    command = retargeter.step(hand(), HOME, 0, engage=True)  # type: ignore[arg-type]
    assert command.target is not None
    assert command.target.distance_to(HOME) == (0.0, 0.0)


def test_lateral_motion_uses_the_lateral_gain() -> None:
    retargeter = Retargeter(CONFIG)
    retargeter.step(hand(image_centre=(0.5, 0.5)), HOME, 0, engage=True)  # type: ignore[arg-type]
    command = retargeter.step(hand(image_centre=(0.7, 0.5)), HOME, 33, engage=True)  # type: ignore[arg-type]
    assert command.target is not None
    # 0.2 of frame width at a 0.2 span = 1.0 palm-span, times 0.10 m.
    assert (command.target.position - HOME.position)[0] == pytest.approx(0.10, abs=1e-12)


def test_depth_motion_uses_the_smaller_depth_gain() -> None:
    """The same 1.0 palm-span of motion moves the tool less in depth, on purpose."""
    retargeter = Retargeter(CONFIG)
    retargeter.step(hand(image_span=0.20), HOME, 0, engage=True)  # type: ignore[arg-type]
    command = retargeter.step(hand(image_span=0.25), HOME, 33, engage=True)  # type: ignore[arg-type]
    assert command.target is not None
    delta_spans = 1 / 0.25 - 1 / 0.20
    expected = CONFIG.depth_scale_m_per_span * delta_spans
    assert (command.target.position - HOME.position)[2] == pytest.approx(expected, abs=1e-12)
    ratio = CONFIG.scale_m_per_span / CONFIG.depth_scale_m_per_span
    assert ratio == pytest.approx(0.10 / 0.035, rel=1e-9)


def test_orientation_is_ignored_unless_asked_for() -> None:
    retargeter = Retargeter(RetargetConfig())
    retargeter.step(hand(), HOME, 0, engage=True)  # type: ignore[arg-type]
    command = retargeter.step(  # type: ignore[arg-type]
        make_hand(rotation=rodrigues((0.0, 0.0, 1.0), 0.8), pinch_ratio=WIDE),
        HOME,
        33,
        engage=True,
    )
    assert command.target is not None
    assert np.abs(command.target.rotation - HOME.rotation).max() == 0.0


def test_orientation_follows_when_enabled() -> None:
    retargeter = Retargeter(RetargetConfig(follow_orientation=True))
    retargeter.step(hand(), HOME, 0, engage=True)  # type: ignore[arg-type]
    turn = rodrigues((0.0, 0.0, 1.0), 0.5)
    command = retargeter.step(  # type: ignore[arg-type]
        make_hand(rotation=turn, pinch_ratio=WIDE), HOME, 33, engage=True
    )
    assert command.target is not None
    assert np.abs(command.target.rotation - turn @ HOME.rotation).max() < 1e-12


def test_a_dropout_holds_both_the_pose_and_the_jaw() -> None:
    retargeter = Retargeter(CONFIG)
    retargeter.step(hand(ratio=MID, image_centre=(0.5, 0.5)), HOME, 0, engage=True)  # type: ignore[arg-type]
    moved = retargeter.step(hand(ratio=MID, image_centre=(0.6, 0.5)), HOME, 33, engage=True)  # type: ignore[arg-type]
    held = retargeter.step(None, HOME, 100, engage=True)

    assert held.target is not None and moved.target is not None
    assert np.abs(held.target.position - moved.target.position).max() == 0.0
    assert held.gripper == moved.gripper
    assert "dropout" in held.reason


def test_re_engaging_rebases_on_the_arms_current_pose() -> None:
    retargeter = Retargeter(CONFIG)
    retargeter.step(hand(image_centre=(0.5, 0.5)), HOME, 0, engage=True)  # type: ignore[arg-type]
    retargeter.step(hand(), HOME, 33, engage=False)  # type: ignore[arg-type]

    elsewhere = Pose(position=np.array([0.10, 0.10, 0.30]), rotation=np.eye(3))
    command = retargeter.step(hand(image_centre=(0.9, 0.9)), elsewhere, 66, engage=True)  # type: ignore[arg-type]
    assert command.target is not None
    assert np.abs(command.target.position - elsewhere.position).max() == 0.0


def test_reset_clears_everything() -> None:
    retargeter = Retargeter(CONFIG)
    retargeter.step(hand(), HOME, 0, engage=True)  # type: ignore[arg-type]
    assert retargeter.clutch is Clutch.ENGAGED
    retargeter.reset()
    assert retargeter.clutch is Clutch.DISENGAGED
    assert retargeter.step(None, HOME, 1, engage=True).target is None


# --- properties ----------------------------------------------------------------
@given(
    ratios=st.lists(st.floats(0.0, 1.5), min_size=1, max_size=60),
    xs=st.lists(st.floats(0.05, 0.95), min_size=1, max_size=60),
)
@settings(max_examples=200, deadline=None)
def test_key_up_never_commands_anything(ratios: list[float], xs: list[float]) -> None:
    """The safety contract: key up means no pose AND no jaw, whatever the hand does."""
    retargeter = Retargeter(CONFIG)
    for index, (ratio, x) in enumerate(zip(ratios, xs, strict=False)):
        command = retargeter.step(  # type: ignore[arg-type]
            hand(ratio=ratio, image_centre=(x, 0.5)), HOME, index * 33, engage=False
        )
        assert command.target is None
        assert command.gripper is None
        assert command.clutch is Clutch.DISENGAGED


@given(
    xs=st.lists(st.floats(0.05, 0.95), min_size=2, max_size=40),
    scale=st.floats(0.01, 0.5),
)
@settings(max_examples=200, deadline=None)
def test_commanded_motion_is_bounded_by_the_gain(xs: list[float], scale: float) -> None:
    """A hand cannot leave the frame, so one drag cannot run the tool away."""
    config = RetargetConfig(scale_m_per_span=scale, depth_scale_m_per_span=scale)
    retargeter = Retargeter(config)
    retargeter.step(hand(image_centre=(xs[0], 0.5)), HOME, 0, engage=True)  # type: ignore[arg-type]
    for index, x in enumerate(xs[1:], start=1):
        command = retargeter.step(  # type: ignore[arg-type]
            hand(image_centre=(x, 0.5)), HOME, index * 33, engage=True
        )
        assert command.target is not None
        assert np.linalg.norm(command.target.position - HOME.position) <= 5.0 * scale + 1e-9


@given(ratio=st.floats(-10.0, 10.0))
@settings(max_examples=200, deadline=None)
def test_jaw_is_always_a_legal_opening(ratio: float) -> None:
    """Whatever the pinch ratio -- including nonsense -- the jaw stays in [0,1]
    and therefore inside the servo's ctrlrange."""
    command = Retargeter(CONFIG).step(hand(ratio=abs(ratio)), HOME, 0, engage=True)  # type: ignore[arg-type]
    assert command.gripper is not None
    assert 0.0 <= command.gripper <= 1.0
    assert GRIPPER_RAD[0] <= gripper_rad(command.gripper) <= GRIPPER_RAD[1]
