"""A UDP datagram of joint targets, from the perception process to the arm.

Two processes, not one, and not because it is tidier. Neither environment can
host both halves without damage: lerobot drags torch and 36 packages into a
repo whose pure suite deliberately has almost none, and mediapipe drags a
second OpenCV distribution into the conda env the club's data collection runs
on. Measured both directions with `uv pip install --dry-run` before splitting.

The split is also just correct. Perception runs at ~20 Hz and the servo bus
wants 50+; coupling them makes the slower one set the rate for both. This is
the same two-loop split that bought 13.3x in sim, with a socket in the seam.

**UDP, and unreliability is the feature.** Every datagram carries a complete
absolute target, so a dropped one costs nothing: the fast loop simply keeps
interpolating toward the previous target, which is what it was doing anyway.
A stream ordered by sequence number lets the receiver discard a late arrival
rather than lurch backwards. Nothing here retries, blocks, or queues, because
a queue of stale robot targets is precisely the wrong thing to act on.

Silence means stop. The receiver's watchdog freezes the arm when packets dry
up, so a crashed or paused sender cannot leave the arm running on a target
that no longer reflects anybody's hand.
"""

from __future__ import annotations

import contextlib
import json
import socket
from dataclasses import dataclass

#: Localhost only. This protocol has no authentication and moves a robot; it
#: must never be reachable off the machine.
HOST = "127.0.0.1"
PORT = 47101

#: The reverse channel, arm -> perception. Not decoration: without it the
#: perception side has to ASSUME a starting pose, and it assumed the sim's.
#: Measured cost of that assumption on our arm: 102 deg on one joint, tool
#: frames 28.7 cm apart, and an operator whose hand did nothing visible while
#: the arm crawled toward a pose nobody asked for. Position is incremental
#: (ADR-002), so "where from" is not a detail, it is the entire frame.
REPORT_PORT = 47102

#: Stop commanding if the newest target is older than this. Two missed frames
#: at 20 Hz perception is 100 ms, so 250 ms distinguishes "a dropped packet"
#: from "the sender is gone" without freezing on ordinary jitter.
STALE_MS = 250.0


@dataclass(frozen=True)
class Target:
    """One absolute joint target. Self-contained, so losing one costs nothing."""

    sequence: int
    joints: dict[str, float]  # degrees, the five body joints
    gripper: float  # openness, 0..1
    engaged: bool  # clutch. False means the operator wants the arm to hold

    def encode(self) -> bytes:
        return json.dumps(
            {"n": self.sequence, "j": self.joints, "g": self.gripper, "e": self.engaged}
        ).encode()

    @staticmethod
    def decode(payload: bytes) -> Target:
        raw = json.loads(payload)
        return Target(
            sequence=int(raw["n"]),
            joints={k: float(v) for k, v in raw["j"].items()},
            gripper=float(raw["g"]),
            engaged=bool(raw["e"]),
        )


@dataclass(frozen=True)
class Report:
    """Where the arm actually is. Sent arm -> perception, so the incremental
    chain can be anchored to the hardware instead of to a guess."""

    joints: dict[str, float]  # degrees, as the servos report them
    gripper: float  # openness, 0..1

    def encode(self) -> bytes:
        return json.dumps({"j": self.joints, "g": self.gripper}).encode()

    @staticmethod
    def decode(payload: bytes) -> Report:
        raw = json.loads(payload)
        return Report({k: float(v) for k, v in raw["j"].items()}, float(raw["g"]))


class _Datagram:
    """Shared non-blocking UDP plumbing. Never blocks, never raises upward."""

    def __init__(self, port: int, host: str, bind: bool) -> None:
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setblocking(False)
        if bind:
            self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._socket.bind((host, port))
        self._address = (host, port)

    def close(self) -> None:
        self._socket.close()


class ReportSender(_Datagram):
    """The arm telling perception where it is. Best effort, like everything."""

    def __init__(self, host: str = HOST, port: int = REPORT_PORT) -> None:
        super().__init__(port, host, bind=False)

    def send(self, joints: dict[str, float], gripper: float) -> None:
        # Perception may not be listening yet, or may have gone. Neither is the
        # arm's problem, and neither may interrupt the control loop.
        with contextlib.suppress(OSError):
            self._socket.sendto(Report(joints, gripper).encode(), self._address)


class ReportReceiver(_Datagram):
    """Perception listening for the arm's actual pose."""

    def __init__(self, host: str = HOST, port: int = REPORT_PORT) -> None:
        super().__init__(port, host, bind=True)
        self.latest: Report | None = None

    def drain(self) -> Report | None:
        while True:
            try:
                payload, _ = self._socket.recvfrom(4096)
            except OSError:
                return self.latest
            try:
                self.latest = Report.decode(payload)
            except (ValueError, KeyError):
                continue

    def wait(self, timeout_s: float, poll_s: float = 0.02) -> Report | None:
        """Block until the arm reports, or give up.

        Used once, at startup. Returning None must be treated as fatal by the
        caller: starting unanchored is the bug this whole channel exists for.
        """
        import time

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            got = self.drain()
            if got is not None:
                return got
            time.sleep(poll_s)
        return None


class Sender:
    """Fire-and-forget. Never blocks the perception loop, never raises at it."""

    def __init__(self, host: str = HOST, port: int = PORT) -> None:
        self._address = (host, port)
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setblocking(False)
        self._sequence = 0
        self.dropped = 0

    def send(self, joints: dict[str, float], gripper: float, *, engaged: bool) -> None:
        self._sequence += 1
        target = Target(self._sequence, joints, gripper, engaged)
        try:
            self._socket.sendto(target.encode(), self._address)
        except OSError:
            # Nobody listening, or the buffer is full. Dropping is correct: the
            # alternative is stalling a vision loop to talk to an absent robot.
            self.dropped += 1

    def close(self) -> None:
        self._socket.close()


class Receiver:
    """Keeps only the newest target. A backlog of stale targets is not data."""

    def __init__(self, host: str = HOST, port: int = PORT) -> None:
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind((host, port))
        self._socket.setblocking(False)
        self.latest: Target | None = None
        self.received = 0
        self.out_of_order = 0

    def drain(self) -> Target | None:
        """Read every waiting datagram, keep the newest, discard the rest."""
        while True:
            try:
                payload, _ = self._socket.recvfrom(4096)
            except BlockingIOError:
                return self.latest
            except OSError:
                return self.latest
            try:
                target = Target.decode(payload)
            except (ValueError, KeyError):
                continue  # a malformed datagram is not a reason to stop a robot
            self.received += 1
            if self.latest is not None and target.sequence <= self.latest.sequence:
                self.out_of_order += 1
                continue  # a late arrival would move the arm backwards
            self.latest = target

    def close(self) -> None:
        self._socket.close()
