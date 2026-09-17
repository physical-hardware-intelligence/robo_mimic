"""Hand -> desired tool pose, through a clutch. The only stateful pure module.

THE CLUTCH (ADR-002)
Pinch to engage, move, release to freeze. Like lifting a mouse off the desk.
Position is INCREMENTAL from wherever you pinched, because a single camera
cannot measure absolute distance. Orientation is drift-free but is also
referenced to the engage pose, so you need not hold your hand at some magic
angle to start.

    DISENGAGED --pinch--> ENGAGED --release--> DISENGAGED
                             |
                       hand lost > grace
                             v
                        DISENGAGED

WHY HYSTERESIS
One threshold chatters: a pinch hovering at the boundary toggles every frame
and the arm stutters. Engage below `pinch_on`, release above `pinch_off`, with
a gap between them.

WHY A GRACE PERIOD
MediaPipe drops a frame now and then. Disengaging on every blip is unusable;
never disengaging is unsafe. Hold the last command for `lost_grace_ms`, then
let go.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .handframe import hand_pose, image_position
from .landmarks import F64, HandLandmarks
from .types import Pose


class Clutch(Enum):
    DISENGAGED = "disengaged"
    ENGAGED = "engaged"


@dataclass(frozen=True, slots=True)
class RetargetConfig:
    """Tunables. Defaults are starting points, not measured optima."""

    #: Pinch ratio (thumb-index over palm span) below which the clutch engages.
    pinch_on: float = 0.35
    #: And above which it releases. Must exceed `pinch_on` -- that gap is the
    #: hysteresis band.
    pinch_off: float = 0.50
    #: Metres of tool motion per palm-span of hand motion. 0.10 means moving
    #: your hand one hand-length moves the tool 10 cm.
    scale_m_per_span: float = 0.10
    #: Hold the last command this long through a dropout before disengaging.
    lost_grace_ms: int = 200
    #: Image height / width. Image y is normalized by height and x by width, so
    #: a non-square frame stretches vertical motion without this.
    aspect: float = 1.0
    #: Map hand rotation to tool rotation at all.
    follow_orientation: bool = True

    def __post_init__(self) -> None:
        if not 0.0 < self.pinch_on < self.pinch_off:
            raise ValueError(
                f"need 0 < pinch_on < pinch_off, got {self.pinch_on}, {self.pinch_off}"
            )
        if self.scale_m_per_span <= 0.0:
            raise ValueError("scale must be positive")
        if self.lost_grace_ms < 0:
            raise ValueError("grace must be non-negative")


@dataclass(frozen=True, slots=True)
class Command:
    """One retargeter output.

    `target` is None whenever the clutch is out. That is not an error: it means
    "do not move", and the safety layer holds position.
    """

    target: Pose | None
    clutch: Clutch
    #: Why this frame did what it did. For the on-screen readout, not control.
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

    Stateful by necessity: a differential mapping needs to remember where you
    pinched. Everything else in the pipeline is a pure function.
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
        """Drop all state. Use on teleport, e-stop, or operator change."""
        self._clutch = Clutch.DISENGAGED
        self._anchor = None
        self._last_command = None
        self._last_seen_ms = None

    def step(
        self, hand: HandLandmarks | None, tool_now: Pose, timestamp_ms: int
    ) -> Command:
        """One frame in, one command out.

        `tool_now` is where the arm actually is. It is read only at the moment of
        engaging, so a drag starts from the arm's true pose rather than from
        wherever the last drag happened to end.
        """
        if hand is None:
            return self._no_hand(timestamp_ms)

        self._last_seen_ms = timestamp_ms
        ratio = hand.pinch_ratio
        pinched = self._pinched(ratio)

        if self._clutch is Clutch.DISENGAGED:
            if not pinched:
                return self._emit(Command(None, Clutch.DISENGAGED, "open hand", ratio))
            self._engage(hand, tool_now)
            return self._emit(Command(tool_now, Clutch.ENGAGED, "engaged", ratio))

        if not pinched:
            self._clutch = Clutch.DISENGAGED
            self._anchor = None
            return self._emit(Command(None, Clutch.DISENGAGED, "released", ratio))

        return self._emit(Command(self._target(hand), Clutch.ENGAGED, "tracking", ratio))

    # --- internals ---------------------------------------------------------
    def _pinched(self, ratio: float) -> bool:
        """Hysteresis: the threshold depends on which state we are already in."""
        if self._clutch is Clutch.ENGAGED:
            return ratio <= self.config.pinch_off
        return ratio <= self.config.pinch_on

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
        position = anchor.tool_position + self.config.scale_m_per_span * delta

        if not self.config.follow_orientation:
            return Pose(position=position, rotation=anchor.tool_rotation)

        # Relative rotation since engaging, applied to the tool's engage pose.
        # `R_hand_now @ R_hand_anchor.T` is the turn the hand has made; composing
        # on the LEFT applies it in the world frame, so the tool turns the way
        # the operator's hand turned, not the way the tool's own axes point.
        turn = hand_pose(hand).rotation @ anchor.hand_rotation.T
        return Pose(position=position, rotation=turn @ anchor.tool_rotation)

    def _no_hand(self, timestamp_ms: int) -> Command:
        if self._clutch is Clutch.DISENGAGED:
            return self._emit(Command(None, Clutch.DISENGAGED, "no hand"))

        elapsed = timestamp_ms - (self._last_seen_ms or timestamp_ms)
        if elapsed <= self.config.lost_grace_ms:
            held = self._last_command
            target = held.target if held is not None else None
            return self._emit(Command(target, Clutch.ENGAGED, f"dropout {elapsed} ms"))

        self._clutch = Clutch.DISENGAGED
        self._anchor = None
        return self._emit(Command(None, Clutch.DISENGAGED, "lost"))

    def _emit(self, command: Command) -> Command:
        self._last_command = command
        return command


__all__ = ["Clutch", "Command", "RetargetConfig", "Retargeter"]
