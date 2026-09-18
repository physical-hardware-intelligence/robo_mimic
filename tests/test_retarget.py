"""The clutch and the jaw: every state transition, plus the gripper mapping.

This module decides whether the arm moves and how hard the gripper squeezes.
Its whole behaviour is a table, so the table is the test.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from robo_mimic.kinematics import GRIPPER_RAD, gripper_openness, gripper_rad
from robo_mimic.retarget import Clutch, RetargetConfig, Retargeter
from robo_mimic.types import Pose

from .conftest_hands import make_hand, rodrigues

HOME = Pose(position=np.array([0.20, 0.0, 0.15]), rotation=np.eye(3))
CONFIG = RetargetConfig(
    scale_m_per_span=0.10, depth_scale_m_per_span=0.035, lost_grace_ms=200
)

MID = 0.525  # (0.25+0.80)/2 -- midway between pinch_closed and pinch_open -> jaw half open
WIDE = 0.95  # above pinch_open -> jaw fully open

#: Measured on a real operator, 2026-09-17: full pinch 0.15 +-0.05, spread hand
#: 0.90 +-0.05. The defaults must saturate the jaw across this WHOLE spread.
OPERATOR_PINCH = (0.10, 0.15, 0.20)
OPERATOR_OPEN = (0.85, 0.90, 0.95)


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
    mapping = config.axis_map
    assert abs(mapping[config.depth_axis, 2]) < abs(mapping[config.lateral_axis, 0])
    assert abs(mapping[config.lateral_axis, 0]) == pytest.approx(config.scale_m_per_span)
    assert abs(mapping[config.depth_axis, 2]) == pytest.approx(
        config.depth_scale_m_per_span
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
        (0.25, 0.0),  # exactly at pinch_closed
        (0.10, 0.0),  # below it -> clamped
        (0.525, 0.5),  # midway
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
    for ratio in np.linspace(0.25, 0.80, 14):
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
    """Lateral hand motion moves the tool LATERALLY -- world y, index 1.

    This test previously asserted index 0, which is the arm's REACH axis. It
    passed, because it was checking what the code did rather than what it should
    do, and so it encoded the axis-mapping bug instead of catching it. Live
    testing caught it: "the arm doesn't move left or right".
    """
    retargeter = Retargeter(CONFIG)
    retargeter.step(hand(image_centre=(0.5, 0.5)), HOME, 0, engage=True)  # type: ignore[arg-type]
    command = retargeter.step(hand(image_centre=(0.7, 0.5)), HOME, 33, engage=True)  # type: ignore[arg-type]
    assert command.target is not None
    delta = command.target.position - HOME.position
    # 0.2 of frame width at a 0.2 span = 1.0 palm-span, times 0.10 m.
    assert delta[1] == pytest.approx(0.10, abs=1e-12)
    assert abs(delta[0]) < 1e-12, "lateral motion must not become reach"
    assert abs(delta[2]) < 1e-12


def test_depth_motion_uses_the_smaller_depth_gain() -> None:
    """Depth moves the tool's REACH -- world x, index 0 -- and less, on purpose.

    Also previously asserted the wrong axis (index 2, up). The gain ratio was
    right; the axis was not.
    """
    retargeter = Retargeter(CONFIG)
    retargeter.step(hand(image_centre=(0.5, 0.5), image_span=0.20), HOME, 0, engage=True)  # type: ignore[arg-type]
    command = retargeter.step(  # type: ignore[arg-type]
        hand(image_centre=(0.5, 0.5), image_span=0.25), HOME, 33, engage=True
    )
    assert command.target is not None
    delta = command.target.position - HOME.position
    # Nearer camera -> larger span -> smaller 1/span -> negated to EXTEND reach.
    expected = -CONFIG.depth_scale_m_per_span * (1 / 0.25 - 1 / 0.20)
    assert delta[0] == pytest.approx(expected, abs=1e-12)
    assert expected > 0.0, "moving the hand nearer must extend the arm"
    assert abs(delta[1]) < 1e-12 and abs(delta[2]) < 1e-12
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


# --- calibration against a real operator ---------------------------------------
@pytest.mark.parametrize("ratio", OPERATOR_PINCH)
def test_the_operators_whole_pinch_spread_fully_closes_the_jaw(ratio: float) -> None:
    """Measured 2026-09-17: a full pinch reads 0.15 +-0.05.

    Every value in that spread must give jaw = 0 exactly. If the thresholds sat
    at the measured edges instead of inside them, a loose pinch at 0.20 would
    leave the jaw 7 percent open -- enough to drop what you thought you gripped.
    """
    command = Retargeter(RetargetConfig()).step(hand(ratio=ratio), HOME, 0, engage=True)  # type: ignore[arg-type]
    assert command.gripper == 0.0


@pytest.mark.parametrize("ratio", OPERATOR_OPEN)
def test_the_operators_whole_open_spread_fully_opens_the_jaw(ratio: float) -> None:
    """Measured 2026-09-17: a spread hand reads 0.90 +-0.05 -> jaw = 1 exactly."""
    command = Retargeter(RetargetConfig()).step(hand(ratio=ratio), HOME, 0, engage=True)  # type: ignore[arg-type]
    assert command.gripper == 1.0


def test_the_thresholds_clear_the_operators_tolerance_with_margin() -> None:
    """The invariant behind the calibration, stated so it cannot silently drift.

    Strict inequalities matter. At 0.20/0.85 the operator's loose pinch sat
    exactly ON the threshold and a float ulp decided whether the jaw shut.
    """
    config = RetargetConfig()
    assert max(OPERATOR_PINCH) < config.pinch_closed, "loose pinch must be strictly inside"
    assert config.pinch_open < min(OPERATOR_OPEN), "low open must be strictly inside"
    margin = min(
        config.pinch_closed - max(OPERATOR_PINCH), min(OPERATOR_OPEN) - config.pinch_open
    )
    # 0.05 is a nominal design margin, not an exact quantity, so compare with
    # slack. Asserting `>= 0.05` exactly failed on 0.85 - 0.80 == 0.04999999999999993
    # -- the very floating-point boundary problem that motivated this margin.
    assert margin == pytest.approx(0.05, abs=1e-9) or margin > 0.05, (
        f"want ~one tolerance-width of margin, got {margin:.4f}"
    )


# --- regression: found by live testing, 2026-09-17 ------------------------------
def test_each_screen_axis_drives_the_right_world_axis() -> None:
    """THE BUG that made left/right do nothing.

    `image_position` returns (screen-x, screen-y, depth); the arm's world frame
    is (reach, lateral, up). An earlier version added them element-wise by
    index, which produced:

        palm RIGHT  -> +5.0 cm of REACH      (should be lateral)
        palm UP     -> -3.8 cm of LATERAL    (should be up)

    Live, that read as "left/right does nothing" -- because left/right was being
    spent on reach, which runs out against the workspace almost at once.
    """
    config = RetargetConfig()
    mapping = config.axis_map

    # screen-x drives lateral (y) and ONLY lateral
    assert mapping[1, 0] != 0.0
    assert mapping[0, 0] == 0.0 and mapping[2, 0] == 0.0
    # screen-y drives up (z) and only up, with a sign flip (image y grows down)
    assert mapping[2, 1] < 0.0
    assert mapping[0, 1] == 0.0 and mapping[1, 1] == 0.0
    # depth drives reach (x) and only reach, negated so nearer = further out
    assert mapping[0, 2] < 0.0
    assert mapping[1, 2] == 0.0 and mapping[2, 2] == 0.0


@pytest.mark.parametrize(
    ("label", "centre", "span", "axis", "sign"),
    [
        ("palm right", (0.6, 0.5), 0.2, 1, +1),
        ("palm left", (0.4, 0.5), 0.2, 1, -1),
        ("palm up", (0.5, 0.4), 0.2, 2, +1),
        ("palm down", (0.5, 0.6), 0.2, 2, -1),
        ("palm closer", (0.5, 0.5), 0.3, 0, +1),
        ("palm further", (0.5, 0.5), 0.15, 0, -1),
    ],
)
def test_hand_motion_moves_the_tool_on_exactly_one_axis(
    label: str, centre: tuple[float, float], span: float, axis: int, sign: int
) -> None:
    """Each of the six directions must move exactly one world axis, the right
    way. A hand on the optical axis keeps the three channels independent."""
    retargeter = Retargeter(RetargetConfig())
    start = make_hand(image_centre=(0.5, 0.5), image_span=0.2)
    retargeter.step(start, HOME, 0, engage=True)  # type: ignore[arg-type]
    moved = make_hand(image_centre=centre, image_span=span)
    command = retargeter.step(moved, HOME, 33, engage=True)

    assert command.target is not None
    delta = command.target.position - HOME.position
    assert np.sign(delta[axis]) == sign, f"{label}: axis {axis} went {delta[axis]:+.4f}"
    for other in range(3):
        if other != axis:
            assert abs(delta[other]) < 1e-12, f"{label}: leaked {delta[other]:+.4f} into {other}"


def test_axis_map_rejects_a_non_permutation() -> None:
    with pytest.raises(ValueError, match="permutation"):
        RetargetConfig(lateral_axis=1, vertical_axis=1, depth_axis=0)
