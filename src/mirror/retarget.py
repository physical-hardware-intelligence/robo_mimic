"""Hand -> desired tool pose + jaw opening. The only stateful pure module.

WHAT DRIVES WHAT (ADR-003)
    a held key      -> clutch. Motion only while it is down.
    hand position   -> tool position, incremental from wherever you engaged.
    thumb-index gap -> jaw opening, continuously.
    hand rotation   -> tool rotation, OFF by default. See below.

    DISENGAGED --key down--> ENGAGED --key up--> DISENGAGED
                                |
                          hand lost > grace
                                v
                           DISENGAGED

WHY THE CLUTCH IS NOT A PINCH
A pinch IS low openness, so one gesture cannot mean both "engage motion" and
"close the jaw" -- you could never approach an object with the gripper open,
which is how grasping works. Splitting them frees the pinch for its natural job.
A held key also cannot false-trigger the way a gesture threshold can, which
makes it a deadman switch rather than a UX preference.

WHY ORIENTATION IS OFF BY DEFAULT
The arm is 5-DOF. Measured: for a position it can reach, only 7-37 percent of
pitch angles are achievable. If hand tilt sets the pitch, most frames have no IK
solution and the arm stutters between solved and unsolvable. Position-only
first; turn `follow_orientation` on once motion is proven in sim.

WHICH SCREEN AXIS DRIVES WHICH WORLD AXIS
-----------------------------------------
`image_position` returns (screen-x, screen-y, depth). The arm's world frame is
(x = reach, radially out from the base; y = lateral; z = up). Those are NOT the
same triple in the same order, and an earlier version added them element-wise
by index, which silently produced:

    palm RIGHT  -> tool +5.0 cm of REACH      (should be lateral)
    palm UP     -> tool -3.8 cm of LATERAL    (should be up)
    palm CLOSER -> -2.4 reach, -6.0 lateral, -4.0 up, all at once

Live, that read as "the gripper follows my pinch, depth sort of works backwards,
and left/right does nothing at all" -- because left/right was being spent on
reach, which runs out against the workspace almost immediately.

`axis_map` states the correspondence explicitly instead of relying on index
coincidence:

    world x (reach)   <-  -depth_scale  * depth      toward the camera = extend
    world y (lateral) <-  +lateral_gain * screen-x
    world z (up)      <-  -lateral_gain * screen-y   image y grows DOWNWARD

Every sign is configurable, because which lateral direction is "right" depends
on where the arm is standing relative to the operator, and that is a fact about
the room rather than about the code.

WHY DEPTH GETS A SMALLER GAIN
`image_position` is `(cx/s, cy*a/s, 1/s)`. Differentiating, lateral error goes
as `dcx/s` but depth error goes as `ds/s**2` -- `s` SQUARED. With `s ~ 0.155`
that is a ~6.5x penalty, and measured end to end:

    camera noise sigma    lateral      depth        ratio
    0.5 levels            0.142 mm     1.041 mm      7.3x
    1.0 levels            0.226 mm     2.331 mm     10.3x
    2.0 levels            0.432 mm    10.822 mm     25.0x

Through the IK that becomes roughly 0.6 deg of joint motion per mm of tool
error. Depth carries an order of magnitude more noise for the same information,
so it gets proportionally less authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .handframe import hand_pose, image_position
from .landmarks import F64, HandLandmarks
from .types import Pose


class Clutch(Enum):
    DISENGAGED = "disengaged"
    ENGAGED = "engaged"


@dataclass(frozen=True, slots=True)
class RetargetConfig:
    """Tunables. Defaults are reasoned starting points, not measured optima."""

    #: Metres of tool motion per palm-span of LATERAL hand motion.
    scale_m_per_span: float = 0.10
    #: Same for DEPTH. Deliberately smaller -- see the module docstring.
    depth_scale_m_per_span: float = 0.035
    #: Pinch ratio (thumb-index gap over palm span) meaning a closed jaw.
    #: CALIBRATED 2026-09-17 against a real operator: a full pinch reads
    #: 0.15 +-0.05 and a spread hand 0.90 +-0.05. These thresholds sit INSIDE
    #: that range, with a margin -- see `pinch_open`.
    pinch_closed: float = 0.25
    #: And a fully open jaw. Must exceed `pinch_closed`.
    #:
    #: WHY INSIDE THE MEASURED RANGE, WITH MARGIN
    #: Three placements were considered against the operator's 0.15/0.90 +-0.05:
    #:
    #:   0.15 / 0.90   AT the edges. A loose pinch (0.20) leaves the jaw 7
    #:                 percent open and a low open (0.85) reaches only 93
    #:                 percent -- both extremes unreachable half the time.
    #:   0.20 / 0.85   AT the tolerance limits. Saturates, but the operator's
    #:                 loose pinch lands EXACTLY on the threshold: measured, a
    #:                 single float ulp (4.27e-17) decided whether the jaw shut.
    #:   0.25 / 0.80   one full tolerance-width of margin beyond the worst case.
    #:                 Chosen.
    #:
    #: The usable band narrows to 0.55 of ratio, about 73 percent of the
    #: operator's 0.75 of travel, with the rest as end deadband. That is the
    #: right trade: reliable full-close IS the grip, and a jaw 7 percent open
    #: when you believe you have gripped drops the object.
    pinch_open: float = 0.80
    #: Hold the last command this long through a tracking dropout.
    lost_grace_ms: int = 200
    #: Image height / width. Image y is normalized by height and x by width.
    aspect: float = 1.0
    #: Map hand rotation to tool rotation. OFF by default -- see the docstring.
    follow_orientation: bool = False

    #: Which world axis each screen axis drives. See the module docstring.
    #: 0 = reach (x), 1 = lateral (y), 2 = up (z).
    lateral_axis: int = 1
    vertical_axis: int = 2
    depth_axis: int = 0
    #: Signs. `vertical` is negative because image y grows downward; `depth` is
    #: negative because the proxy is `1/span`, which SHRINKS as the hand nears
    #: the camera -- so a negative sign makes "hand toward camera" extend the arm.
    lateral_sign: float = 1.0
    vertical_sign: float = -1.0
    depth_sign: float = -1.0

    def __post_init__(self) -> None:
        if self.scale_m_per_span <= 0.0 or self.depth_scale_m_per_span <= 0.0:
            raise ValueError("scales must be positive")
        if not 0.0 <= self.pinch_closed < self.pinch_open:
            raise ValueError(
                f"need 0 <= pinch_closed < pinch_open, "
                f"got {self.pinch_closed}, {self.pinch_open}"
            )
        if self.lost_grace_ms < 0:
            raise ValueError("grace must be non-negative")
        axes = (self.lateral_axis, self.vertical_axis, self.depth_axis)
        if sorted(axes) != [0, 1, 2]:
            raise ValueError(
                f"lateral/vertical/depth axes must be a permutation of 0,1,2; got {axes}"
            )

    @property
    def axis_map(self) -> F64:
        """3x3 mapping proxy deltas to world deltas, in metres per palm-span.

        `world_delta = axis_map @ (screen-x, screen-y, depth)`. A matrix rather
        than three gains because the correspondence is a permutation with signs,
        and writing it as one makes the thing that was wrong impossible to get
        wrong silently.
        """
        matrix = np.zeros((3, 3))
        matrix[self.lateral_axis, 0] = self.lateral_sign * self.scale_m_per_span
        matrix[self.vertical_axis, 1] = self.vertical_sign * self.scale_m_per_span
        matrix[self.depth_axis, 2] = self.depth_sign * self.depth_scale_m_per_span
        return matrix


@dataclass(frozen=True, slots=True)
class Command:
    """One retargeter output.

    `target` and `gripper` are BOTH None whenever the clutch is out. That is not
    an error: it means "do not move", and the safety layer holds position. The
    jaw is gated too -- a deadman that still lets the gripper move is not one.
    """

    target: Pose | None
    #: Jaw opening in [0, 1]. 0 = closed, 1 = open. Unit-free on purpose;
    #: `kinematics.gripper_rad` is the only place that knows servo units.
    gripper: float | None
    clutch: Clutch
    #: Why this frame did what it did. For the readout, not for control.
    reason: str
    pinch_ratio: float | None = None


@dataclass(frozen=True, slots=True)
class _Anchor:
    """Captured at engage. The zero point for this drag."""

    hand_position: F64
    hand_rotation: F64
    tool_position: F64
    tool_rotation: F64


class Retargeter:
    """Feed it frames in order; it emits commands.

    Stateful by necessity: an incremental mapping must remember where you
    engaged. Everything else in the pipeline is a pure function.
    """

    def __init__(self, config: RetargetConfig | None = None) -> None:
        self.config = config or RetargetConfig()
        self._clutch = Clutch.DISENGAGED
        self._anchor: _Anchor | None = None
        self._last_command: Command | None = None
        self._last_seen_ms: int | None = None

    @property
    def clutch(self) -> Clutch:
        return self._clutch

    def reset(self) -> None:
        """Drop all state. Use on e-stop, teleport, or operator change."""
        self._clutch = Clutch.DISENGAGED
        self._anchor = None
        self._last_command = None
        self._last_seen_ms = None

    def step(
        self,
        hand: HandLandmarks | None,
        tool_now: Pose,
        timestamp_ms: int,
        *,
        engage: bool,
    ) -> Command:
        """One frame in, one command out.

        `engage` is the deadman: True while the operator holds the clutch. A
        plain boolean, so the source can be a key, a pedal, or a hardware switch
        without this module caring.

        `tool_now` is where the arm actually is. It is read only at the moment of
        engaging, so a drag starts from the arm's true pose rather than from
        wherever the previous drag happened to end.
        """
        if hand is None:
            return self._no_hand(timestamp_ms, engage)

        self._last_seen_ms = timestamp_ms
        ratio = hand.pinch_ratio
        jaw = self._jaw(ratio)

        if not engage:
            was_engaged = self._clutch is Clutch.ENGAGED
            self._clutch = Clutch.DISENGAGED
            self._anchor = None
            reason = "released" if was_engaged else "clutch out"
            return self._emit(Command(None, None, Clutch.DISENGAGED, reason, ratio))

        if self._clutch is Clutch.DISENGAGED:
            self._engage(hand, tool_now)
            return self._emit(Command(tool_now, jaw, Clutch.ENGAGED, "engaged", ratio))

        return self._emit(Command(self._target(hand), jaw, Clutch.ENGAGED, "tracking", ratio))

    # --- internals ---------------------------------------------------------
    def _jaw(self, ratio: float) -> float:
        """Pinch ratio -> jaw opening in [0,1], clamped.

        Linear between `pinch_closed` and `pinch_open`. A RATIO rather than
        metres because dividing by palm span removes the operator's hand size,
        so one setting fits every operator.
        """
        lo, hi = self.config.pinch_closed, self.config.pinch_open
        return float(np.clip((ratio - lo) / (hi - lo), 0.0, 1.0))

    def _engage(self, hand: HandLandmarks, tool_now: Pose) -> None:
        self._clutch = Clutch.ENGAGED
        self._anchor = _Anchor(
            hand_position=image_position(hand, self.config.aspect),
            hand_rotation=hand_pose(hand).rotation,
            tool_position=tool_now.position,
            tool_rotation=tool_now.rotation,
        )

    def _target(self, hand: HandLandmarks) -> Pose:
        anchor = self._anchor
        if anchor is None:  # pragma: no cover - guarded by the state machine
            raise RuntimeError("engaged with no anchor")

        delta = image_position(hand, self.config.aspect) - anchor.hand_position
        position = anchor.tool_position + self.config.axis_map @ delta

        if not self.config.follow_orientation:
            return Pose(position=position, rotation=anchor.tool_rotation)

        # `R_now @ R_anchor.T` is the turn the hand has made since engaging.
        # Composing on the LEFT applies it in the world frame, so the tool turns
        # the way the operator's hand turned, not the way the tool's axes point.
        turn = hand_pose(hand).rotation @ anchor.hand_rotation.T
        return Pose(position=position, rotation=turn @ anchor.tool_rotation)

    def _no_hand(self, timestamp_ms: int, engage: bool) -> Command:
        if self._clutch is Clutch.DISENGAGED:
            return self._emit(Command(None, None, Clutch.DISENGAGED, "no hand"))

        if not engage:
            self._clutch = Clutch.DISENGAGED
            self._anchor = None
            return self._emit(Command(None, None, Clutch.DISENGAGED, "released"))

        elapsed = timestamp_ms - (self._last_seen_ms or timestamp_ms)
        if elapsed <= self.config.lost_grace_ms:
            held = self._last_command
            return self._emit(
                Command(
                    held.target if held else None,
                    held.gripper if held else None,
                    Clutch.ENGAGED,
                    f"dropout {elapsed} ms",
                )
            )

        self._clutch = Clutch.DISENGAGED
        self._anchor = None
        return self._emit(Command(None, None, Clutch.DISENGAGED, "lost"))

    def _emit(self, command: Command) -> Command:
        self._last_command = command
        return command


__all__ = ["Clutch", "Command", "RetargetConfig", "Retargeter"]
