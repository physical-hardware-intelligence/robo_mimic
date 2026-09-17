"""The clutch: every state transition, enumerated.

This is the module that decides whether the arm moves. Its whole behaviour is a
four-row table, so the table is the test.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mirror.retarget import Clutch, RetargetConfig, Retargeter
from mirror.types import Pose

from .conftest_hands import make_hand, rodrigues

HOME = Pose(position=np.array([0.20, 0.0, 0.15]), rotation=np.eye(3))
CONFIG = RetargetConfig(pinch_on=0.35, pinch_off=0.50, scale_m_per_span=0.10, lost_grace_ms=200)

OPEN = 0.90  # well above pinch_off
SHUT = 0.20  # well below pinch_on
BAND = 0.42  # inside the hysteresis band: engages nothing, releases nothing


def pinched(ratio: float, **kwargs: object) -> object:
    return make_hand(pinch_ratio=ratio, **kwargs)  # type: ignore[arg-type]


# --- config validation --------------------------------------------------------
@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"pinch_on": 0.6, "pinch_off": 0.4}, "pinch_on < pinch_off"),
        ({"pinch_on": 0.0}, "pinch_on < pinch_off"),
        ({"pinch_on": -0.1}, "pinch_on < pinch_off"),
        ({"scale_m_per_span": 0.0}, "scale must be positive"),
        ({"lost_grace_ms": -1}, "grace must be non-negative"),
    ],
)
def test_bad_config_is_rejected(kwargs: dict[str, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        RetargetConfig(**kwargs)  # type: ignore[arg-type]


# --- THE STATE TABLE ----------------------------------------------------------
# (start state, hand?, pinch ratio, ms since last seen) -> (end state, commands?)
TABLE = [
    ("disengaged, open hand", Clutch.DISENGAGED, OPEN, 0, Clutch.DISENGAGED, False),
    ("disengaged, pinch", Clutch.DISENGAGED, SHUT, 0, Clutch.ENGAGED, True),
    ("disengaged, in band", Clutch.DISENGAGED, BAND, 0, Clutch.DISENGAGED, False),
    ("disengaged, no hand", Clutch.DISENGAGED, None, 0, Clutch.DISENGAGED, False),
    ("engaged, still pinched", Clutch.ENGAGED, SHUT, 0, Clutch.ENGAGED, True),
    ("engaged, in band", Clutch.ENGAGED, BAND, 0, Clutch.ENGAGED, True),
    ("engaged, released", Clutch.ENGAGED, OPEN, 0, Clutch.DISENGAGED, False),
    ("engaged, brief dropout", Clutch.ENGAGED, None, 100, Clutch.ENGAGED, True),
    ("engaged, dropout at limit", Clutch.ENGAGED, None, 200, Clutch.ENGAGED, True),
    ("engaged, dropout past limit", Clutch.ENGAGED, None, 201, Clutch.DISENGAGED, False),
]


@pytest.mark.parametrize(
    ("name", "start", "ratio", "elapsed", "expect_state", "expect_target"),
    TABLE,
    ids=[row[0] for row in TABLE],
)
def test_every_clutch_transition(
    name: str,
    start: Clutch,
    ratio: float | None,
    elapsed: int,
    expect_state: Clutch,
    expect_target: bool,
) -> None:
    retargeter = Retargeter(CONFIG)
    time_ms = 1000

    if start is Clutch.ENGAGED:
        retargeter.step(pinched(SHUT), HOME, time_ms)  # type: ignore[arg-type]
        assert retargeter.clutch is Clutch.ENGAGED

    hand = None if ratio is None else pinched(ratio)
    command = retargeter.step(hand, HOME, time_ms + elapsed)  # type: ignore[arg-type]

    assert command.clutch is expect_state, name
    assert (command.target is not None) is expect_target, name
    assert retargeter.clutch is expect_state, name


def test_hysteresis_band_does_nothing_in_either_direction() -> None:
    """The point of two thresholds: the band is sticky, not a third state."""
    retargeter = Retargeter(CONFIG)
    assert retargeter.step(pinched(BAND), HOME, 0).clutch is Clutch.DISENGAGED  # type: ignore[arg-type]
    retargeter.step(pinched(SHUT), HOME, 1)  # type: ignore[arg-type]
    assert retargeter.step(pinched(BAND), HOME, 2).clutch is Clutch.ENGAGED  # type: ignore[arg-type]


def test_a_ratio_oscillating_in_the_band_never_chatters() -> None:
    """Without hysteresis this toggles every frame and the arm stutters."""
    retargeter = Retargeter(CONFIG)
    retargeter.step(pinched(SHUT), HOME, 0)  # type: ignore[arg-type]
    states = [
        retargeter.step(pinched(0.36 + 0.10 * (i % 2)), HOME, i).clutch  # type: ignore[arg-type]
        for i in range(1, 21)
    ]
    assert set(states) == {Clutch.ENGAGED}


# --- what it commands ---------------------------------------------------------
def test_engaging_commands_exactly_where_the_arm_already_is() -> None:
    """No jump on engage. The first command must be a no-op."""
    retargeter = Retargeter(CONFIG)
    command = retargeter.step(pinched(SHUT), HOME, 0)  # type: ignore[arg-type]
    assert command.target is not None
    metres, degrees = command.target.distance_to(HOME)
    assert metres == 0.0
    assert degrees == 0.0


def test_moving_the_hand_moves_the_tool_by_scale_times_the_offset() -> None:
    retargeter = Retargeter(CONFIG)
    retargeter.step(pinched(SHUT, image_centre=(0.5, 0.5)), HOME, 0)  # type: ignore[arg-type]
    command = retargeter.step(pinched(SHUT, image_centre=(0.7, 0.5)), HOME, 33)  # type: ignore[arg-type]

    assert command.target is not None
    # 0.2 of frame width at a 0.2 span = 1.0 palm-span, times 0.10 m = 10 cm.
    delta = command.target.position - HOME.position
    assert delta[0] == pytest.approx(0.10, abs=1e-12)
    assert delta[1] == pytest.approx(0.0, abs=1e-12)


def test_the_tool_follows_the_hands_rotation() -> None:
    retargeter = Retargeter(CONFIG)
    retargeter.step(pinched(SHUT), HOME, 0)  # type: ignore[arg-type]
    turn = rodrigues((0.0, 0.0, 1.0), 0.5)
    command = retargeter.step(  # type: ignore[arg-type]
        make_hand(rotation=turn, pinch_ratio=SHUT), HOME, 33
    )
    assert command.target is not None
    assert np.abs(command.target.rotation - turn @ HOME.rotation).max() < 1e-12


def test_orientation_can_be_switched_off() -> None:
    retargeter = Retargeter(RetargetConfig(follow_orientation=False))
    retargeter.step(pinched(SHUT), HOME, 0)  # type: ignore[arg-type]
    command = retargeter.step(  # type: ignore[arg-type]
        make_hand(rotation=rodrigues((1.0, 0.0, 0.0), 1.0), pinch_ratio=SHUT), HOME, 33
    )
    assert command.target is not None
    assert np.abs(command.target.rotation - HOME.rotation).max() == 0.0


def test_a_dropout_holds_the_last_target_rather_than_inventing_one() -> None:
    retargeter = Retargeter(CONFIG)
    retargeter.step(pinched(SHUT, image_centre=(0.5, 0.5)), HOME, 0)  # type: ignore[arg-type]
    moved = retargeter.step(pinched(SHUT, image_centre=(0.6, 0.5)), HOME, 33)  # type: ignore[arg-type]
    held = retargeter.step(None, HOME, 100)

    assert held.target is not None and moved.target is not None
    assert np.abs(held.target.position - moved.target.position).max() == 0.0
    assert "dropout" in held.reason


def test_re_engaging_rebases_on_the_arms_current_pose() -> None:
    """Release, move the arm elsewhere, re-pinch: no jump back to the old anchor."""
    retargeter = Retargeter(CONFIG)
    retargeter.step(pinched(SHUT, image_centre=(0.5, 0.5)), HOME, 0)  # type: ignore[arg-type]
    retargeter.step(pinched(OPEN), HOME, 33)  # type: ignore[arg-type]

    elsewhere = Pose(position=np.array([0.10, 0.10, 0.30]), rotation=np.eye(3))
    command = retargeter.step(pinched(SHUT, image_centre=(0.9, 0.9)), elsewhere, 66)  # type: ignore[arg-type]
    assert command.target is not None
    assert np.abs(command.target.position - elsewhere.position).max() == 0.0


def test_reset_clears_everything() -> None:
    retargeter = Retargeter(CONFIG)
    retargeter.step(pinched(SHUT), HOME, 0)  # type: ignore[arg-type]
    assert retargeter.clutch is Clutch.ENGAGED
    retargeter.reset()
    assert retargeter.clutch is Clutch.DISENGAGED
    assert retargeter.step(None, HOME, 1).target is None


# --- properties ----------------------------------------------------------------
@given(
    ratios=st.lists(st.floats(0.05, 1.5), min_size=1, max_size=60),
    xs=st.lists(st.floats(0.05, 0.95), min_size=1, max_size=60),
)
@settings(max_examples=200, deadline=None)
def test_a_disengaged_clutch_never_commands_motion(
    ratios: list[float], xs: list[float]
) -> None:
    """The safety-relevant half of the contract: open hand, no command.

    Whatever the hand does, if the clutch is out the target is None. The safety
    layer turns that into "hold position".
    """
    retargeter = Retargeter(CONFIG)
    for index, (ratio, x) in enumerate(zip(ratios, xs, strict=False)):
        command = retargeter.step(  # type: ignore[arg-type]
            pinched(ratio, image_centre=(x, 0.5)), HOME, index * 33
        )
        if command.clutch is Clutch.DISENGAGED:
            assert command.target is None


@given(
    xs=st.lists(st.floats(0.05, 0.95), min_size=2, max_size=40),
    scale=st.floats(0.01, 0.5),
)
@settings(max_examples=200, deadline=None)
def test_commanded_motion_is_bounded_by_the_scale(xs: list[float], scale: float) -> None:
    """A hand cannot leave the frame, so the tool cannot run away.

    Screen x lives in [0,1] and the span is fixed at 0.2, so the proxy spans at
    most 5 palm-spans. Times `scale`, that bounds how far one drag can move the
    tool -- a useful thing to know before Phase 4 adds hard limits.
    """
    retargeter = Retargeter(RetargetConfig(scale_m_per_span=scale))
    retargeter.step(pinched(SHUT, image_centre=(xs[0], 0.5)), HOME, 0)  # type: ignore[arg-type]
    for index, x in enumerate(xs[1:], start=1):
        command = retargeter.step(  # type: ignore[arg-type]
            pinched(SHUT, image_centre=(x, 0.5)), HOME, index * 33
        )
        assert command.target is not None
        assert np.linalg.norm(command.target.position - HOME.position) <= 5.0 * scale + 1e-9
