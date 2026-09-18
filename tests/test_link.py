"""The process link. Real sockets on loopback, no arm, no camera."""

from __future__ import annotations

import time

import pytest

from robo_mimic.link import Receiver, Sender, Target

JOINTS = {"shoulder_pan": 1.5, "shoulder_lift": -100.0, "elbow_flex": 90.0,
          "wrist_flex": 40.0, "wrist_roll": -62.5}
PORT = 47199  # not the default, so a real session cannot collide with a test


def settled(receiver: Receiver, timeout_s: float = 1.0) -> Target | None:
    """Drain until something arrives or we give up.

    Loopback UDP is fast but not synchronous: draining in the same microsecond
    as the send legitimately finds nothing. The real receiver polls at 50 Hz so
    a datagram that lands 50 us late is simply picked up next tick. A test that
    drains exactly once is testing the scheduler, not the code.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        got = receiver.drain()
        if got is not None:
            return got
        time.sleep(0.002)
    return None


@pytest.fixture
def pair():
    receiver = Receiver(port=PORT)
    sender = Sender(port=PORT)
    yield sender, receiver
    sender.close()
    receiver.close()


class TestEncoding:
    def test_round_trip_preserves_every_field(self) -> None:
        original = Target(7, JOINTS, 0.42, True)
        decoded = Target.decode(original.encode())
        assert decoded.sequence == 7
        assert decoded.joints == pytest.approx(JOINTS)
        assert decoded.gripper == pytest.approx(0.42)
        assert decoded.engaged is True

    def test_a_target_fits_in_one_datagram(self) -> None:
        """Fragmented UDP is lost UDP. Well under the 1500-byte path MTU."""
        assert len(Target(999999, JOINTS, 1.0, True).encode()) < 512


class TestDelivery:
    def test_a_sent_target_arrives(self, pair) -> None:
        sender, receiver = pair
        sender.send(JOINTS, 0.3, engaged=True)
        got = settled(receiver)
        assert got is not None
        assert got.joints == pytest.approx(JOINTS)
        assert got.engaged is True

    def test_drain_keeps_the_newest_not_the_first(self, pair) -> None:
        """A backlog of stale robot targets is not data to work through."""
        sender, receiver = pair
        for openness in (0.1, 0.2, 0.9):
            sender.send(JOINTS, openness, engaged=True)
        time.sleep(0.05)          # let all three land, so drain has a backlog
        got = receiver.drain()
        assert got is not None
        assert got.gripper == pytest.approx(0.9)
        assert receiver.received == 3

    def test_a_late_arrival_is_discarded_not_applied(self, pair) -> None:
        """Out of order, a stale target would drive the arm backwards."""
        sender, receiver = pair
        sender.send(JOINTS, 0.9, engaged=True)
        assert settled(receiver) is not None
        stale = Target(0, JOINTS, 0.1, True)
        sender._socket.sendto(stale.encode(), ("127.0.0.1", PORT))
        time.sleep(0.05)
        got = receiver.drain()
        assert got is not None
        assert got.gripper == pytest.approx(0.9)
        assert receiver.out_of_order == 1

    def test_nothing_sent_means_nothing_to_act_on(self, pair) -> None:
        _, receiver = pair
        assert receiver.drain() is None

    def test_a_malformed_datagram_does_not_stop_the_stream(self, pair) -> None:
        """Garbage on the wire must not be able to halt a running robot."""
        sender, receiver = pair
        sender._socket.sendto(b"not json at all", ("127.0.0.1", PORT))
        sender.send(JOINTS, 0.55, engaged=True)
        got = settled(receiver)
        assert got is not None
        assert got.gripper == pytest.approx(0.55)


class TestSenderNeverBlocksPerception:
    def test_sending_into_the_void_is_not_an_error(self) -> None:
        """No receiver bound. The vision loop must not care."""
        sender = Sender(port=47198)
        try:
            for _ in range(50):
                sender.send(JOINTS, 0.5, engaged=False)
        finally:
            sender.close()

    def test_the_socket_is_non_blocking(self) -> None:
        sender = Sender(port=47197)
        try:
            assert sender._socket.getblocking() is False
        finally:
            sender.close()
