"""The real SO-101. The only file in this repo that can move something heavy.

    python scripts/arm.py check          # read-only. Touches no torque.
    python scripts/arm.py hold           # energise and hold still. The e-stop test.
    python scripts/arm.py jog --joint wrist_roll --deg 10

Run it with the `phi` conda env's python: that is where lerobot, the Feetech
SDK and this arm's calibration already live, all of them proven on this rig.
Adding lerobot to our own lock would drag a training stack into a repo whose
point is that 218 tests run with no camera and no arm.

    ~/miniforge3/envs/phi/bin/python scripts/arm.py check

STOP SEMANTICS, because getting this wrong breaks hardware:

  freeze  Goal_Position := Present_Position, torque stays ON. The arm holds
          where it is. This is the response to every NORMAL stop -- clutch
          released, hand lost, you pressed q.
  limp    torque OFF. The arm FALLS. Correct only when something is jammed or
          someone is caught, and never as a routine exit, which is why it is a
          separate key and not the default.

Exit always parks: freeze, then walk home slowly, then limp. In that order.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from robo_mimic.arm import (  # noqa: E402
    FOLDED,
    GRIPPER,
    MAX_TEMP_C,
    MIN_VOLTS,
    READY,
    check_calibration,
    entry_pose,
    report_pose,
    stale_goal_deg,
)
from robo_mimic.kinematics.limits import JOINTS, clamp_deg  # noqa: E402

#: phi's follower. Overridable, never guessed: picking the LEADER by accident
#: would drive the arm you are supposed to be holding.
DEFAULT_PORT = os.environ.get("FOLLOWER_PORT", "/dev/tty.usbmodem5B7B0096441")
DEFAULT_CALIBRATION = Path(
    os.environ.get(
        "SO101_CALIBRATION",
        "/Volumes/Crucial_X9/Projects/phi/configs/calibration/robots/so_follower/phi_follower.json",
    )
)
RESOLUTION = 4095  # sts3215 ticks per turn, minus one

GREEN, RED, YELLOW, DIM, OFF = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"
_failures: list[str] = []


def line(ok: bool | None, label: str, detail: str, fix: str = "") -> None:
    mark = f"{GREEN}ok{OFF}" if ok else (f"{YELLOW}--{OFF}" if ok is None else f"{RED}NO{OFF}")
    print(f"  [{mark}] {label:<24} {detail}")
    if ok is False:
        _failures.append(label)
        if fix:
            print(f"       {DIM}{fix}{OFF}")


def open_bus(port: str, calibration_path: Path):
    """Open the serial port. Does NOT touch torque -- `connect` only pings."""
    from lerobot.motors import Motor, MotorCalibration, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus

    raw = json.loads(calibration_path.read_text())
    body = MotorNormMode.DEGREES
    bus = FeetechMotorsBus(
        port=port,
        motors={
            "shoulder_pan": Motor(1, "sts3215", body),
            "shoulder_lift": Motor(2, "sts3215", body),
            "elbow_flex": Motor(3, "sts3215", body),
            "wrist_flex": Motor(4, "sts3215", body),
            "wrist_roll": Motor(5, "sts3215", body),
            "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
        },
        calibration={n: MotorCalibration(**c) for n, c in raw.items()},
    )
    bus.connect()
    return bus, raw


def cmd_check(args: argparse.Namespace) -> int:
    """Everything that must be true before the arm is allowed to move."""
    print(f"\n  {DIM}port {args.port}{OFF}\n")

    if not DEFAULT_CALIBRATION.exists() and not args.calibration.exists():
        line(False, "calibration file", f"missing: {args.calibration}",
             "point --calibration at the follower's json")
        return 1

    bus, raw = open_bus(args.port, args.calibration)
    try:
        line(True, "serial port", f"open, {len(bus.motors)} motors pinged")
        line(bus.is_calibrated, "calibration", "motors agree with the file"
             if bus.is_calibrated else "motors DISAGREE with the file",
             "lerobot would prompt to recalibrate; do that deliberately, not mid-run")

        for joint, calibrated, ours, subset in check_calibration(raw, RESOLUTION):
            if joint == "gripper":
                continue
            line(subset, f"travel {joint}",
                 f"we command {ours:6.2f} of {calibrated:6.2f} deg"
                 f"  ({calibrated - ours:+.2f} deg margin)",
                 "our limits exceed the arm's calibrated travel: we would hit a stop")

        torque = bus.sync_read("Torque_Enable", normalize=False)
        energised = [n for n in bus.motors if torque[n]]
        state = "ENERGISED: " + ", ".join(energised) if energised else "all off, arm is limp"
        line(True, "torque at rest", state)

        present_t = bus.sync_read("Present_Position", normalize=False)
        goal_t = bus.sync_read("Goal_Position", normalize=False)
        stale = stale_goal_deg(present_t, goal_t, RESOLUTION)
        line(stale <= args.max_stale, "stale goal", f"{stale:.2f} deg worst joint",
             f"energising would THROW the arm up to {stale:.1f} deg. Fix it with "
             "`arm.py align`, which writes goal := present with the torque still off")
        if stale > args.max_stale:
            for n in bus.motors:
                gap = (goal_t[n] - present_t[n]) * 360.0 / RESOLUTION
                if abs(gap) > 1.0:
                    print(f"       {DIM}{n:<15}{present_t[n]:>6} -> {goal_t[n]:<6} "
                          f"{gap:+8.2f} deg{OFF}")

        volts = bus.sync_read("Present_Voltage", normalize=False)
        temps = bus.sync_read("Present_Temperature", normalize=False)
        low = min(volts.values()) / 10
        hot = max(temps.values())
        line(low >= MIN_VOLTS, "supply", f"{low:.1f} V", "servos are browning out")
        line(hot <= MAX_TEMP_C, "temperature", f"{hot} C peak", "let it cool")

        present_d = bus.sync_read("Present_Position")
        body = {n: present_d[n] for n in JOINTS}
        outside = [r for r in report_pose(body) if not r.inside]
        _, correction = entry_pose(body)
        line(
            correction <= args.max_entry,
            "rest pose vs limits",
            "inside the envelope" if not outside else
            ", ".join(f"{r.joint} {r.present_deg:+.2f} is {r.excursion_deg:.2f} deg out"
                      for r in outside),
            f"the first clamped command would move it {correction:.1f} deg unbidden",
        )
        if outside and correction <= args.max_entry:
            print(f"       {DIM}power-up will correct {correction:.2f} deg. Small, but it is "
                  f"motion you did not ask for{OFF}")

        print(f"\n  {DIM}{'joint':<15}{'deg':>9}{'ticks':>8}{'V':>7}{'C':>5}{OFF}")
        for n in bus.motors:
            print(f"  {n:<15}{present_d[n]:>9.2f}{present_t[n]:>8}"
                  f"{volts[n]/10:>7.1f}{temps[n]:>5}")
    finally:
        # Leave the arm EXACTLY as found. Disabling torque on an energised arm
        # would drop it, which is the opposite of a safe read-only check.
        bus.disconnect(disable_torque=False)

    if _failures:
        print(f"\n  {RED}{len(_failures)} blocker(s): {', '.join(_failures)}{OFF}")
        print(f"  {DIM}nothing will be energised until these are clear{OFF}\n")
        return 1
    print(f"\n  {GREEN}clear to energise{OFF}\n")
    return 0


class Arm:
    """Energised control of the follower, with the stop semantics up top.

    Constructed only after `check` has passed. Every path out of here parks the
    arm: freeze, walk home, then limp -- in that order, including on exception.
    """

    def __init__(self, bus, rate_hz: float = 50.0, speed_deg_s: float = 15.0) -> None:
        self.bus = bus
        self.dt = 1.0 / rate_hz
        self.speed = speed_deg_s
        self.limp_now = False

        # Where to leave the arm, and it is NOT always where we found it.
        #
        # Park ends in torque off, so the parking pose must be one the arm can
        # hold with NO torque at all. If we found it limp, then wherever it was
        # sagging is by definition such a pose. If we found it ENERGISED -- a
        # previous session left torque on, which happens -- it may be holding
        # itself somewhere it cannot hold itself, and returning there before
        # cutting torque would drop it from height. Fold it down instead.
        found_energised = any(self.bus.sync_read("Torque_Enable", normalize=False).values())
        self.home = self.present()
        if found_energised:
            self.home = {**self.home, **FOLDED}
            print(f"  {DIM}found energised, so parking will FOLD it rather than "
                  f"return it to a pose it may not hold unpowered{OFF}")

    def present(self) -> dict[str, float]:
        return dict(self.bus.sync_read("Present_Position"))

    def energise(self) -> None:
        """Torque on WITHOUT motion: aim at where it already is, then enable.

        Enabling torque makes each servo drive to whatever Goal_Position it
        still holds from a previous session. Writing the present position first
        makes that target a no-op, so power-up cannot snap. `check` measures the
        gap too, but this removes it rather than just reporting it.
        """
        self.bus.sync_write("Goal_Position", self.present())
        self.bus.enable_torque()

    def freeze(self) -> None:
        """Hold position, torque ON. The normal stop."""
        self.bus.sync_write("Goal_Position", self.present())

    def limp(self) -> None:
        """Torque OFF. The arm FALLS. Only for a jam or a trapped hand."""
        self.limp_now = True
        self.bus.disable_torque()

    def glide(self, target: dict[str, float], label: str = "") -> None:
        """Walk to `target` at `speed_deg_s`, never faster, interpolating.

        A single sync_write of a far-away goal would make the servos sprint at
        whatever their internal profile allows. Stepping the setpoint is how
        speed gets bounded at all -- the same reason the sim loop interpolates.
        """
        start = self.present()
        moving = {k: v for k, v in target.items() if k in start}
        worst = max((abs(moving[k] - start[k]) for k in moving), default=0.0)
        if worst < 0.05:
            return
        steps = max(1, int(worst / self.speed / self.dt))
        if label:
            print(f"  {label}: {worst:.2f} deg over {steps * self.dt:.1f} s")
        for i in range(1, steps + 1):
            f = i / steps
            self.bus.sync_write(
                "Goal_Position", {k: start[k] + f * (moving[k] - start[k]) for k in moving}
            )
            time.sleep(self.dt)

    def park(self) -> None:
        """The only way out. Freeze, walk home, let go, then align.

        The align at the end is not cosmetic. Torque off lets the arm sag away
        from its last goal, and that gap is exactly what makes the NEXT power-up
        snap. Leaving goal == where the limp arm actually settled means whoever
        runs this next finds a safe arm rather than a loaded spring.
        """
        if self.limp_now:
            align(self.bus)
            return
        try:
            self.freeze()
            self.glide(self.home, "parking")
        finally:
            self.bus.disable_torque()
            time.sleep(0.4)  # let gravity finish before recording where it landed
            align(self.bus)
            print("  parked, torque off, goal aligned to where it settled")


def align(bus) -> float:
    """Goal_Position := Present_Position. Torque untouched, nothing moves.

    The cure for a stale goal left behind by any other process -- a lerobot
    rollout, a teleop session, a crash. Writing the goal register while torque
    is off cannot move the arm; it only disarms the snap that enabling torque
    would otherwise cause.
    """
    present = bus.sync_read("Present_Position", normalize=False)
    goal = bus.sync_read("Goal_Position", normalize=False)
    before = stale_goal_deg(present, goal, RESOLUTION)
    bus.sync_write("Goal_Position", present, normalize=False)
    return before


def _keys():
    """Non-blocking single keypresses, restoring the terminal on the way out."""
    import select
    import termios
    import tty

    fd = sys.stdin.fileno()
    if not sys.stdin.isatty():
        yield from iter(lambda: "", None)
        return
    saved = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while True:
            ready, _, _ = select.select([sys.stdin], [], [], 0)
            yield sys.stdin.read(1) if ready else ""
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)


def _park_on_sigterm() -> None:
    """Make SIGTERM take the same exit as ctrl-c.

    Default SIGTERM kills the process outright: torque stays on, the arm holds
    a pose nobody is commanding, and the next session finds a stale goal. A
    `pkill` did exactly that here and left the arm energised for ten minutes.
    Raising KeyboardInterrupt routes it through the normal park.
    """
    import signal

    def handler(_signum: int, _frame: object) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGHUP, handler)


def _energised_session(args: argparse.Namespace, body: Callable[[Arm], int]) -> int:
    """Shared bring-up: re-run the gate, energise, run `body`, always park."""
    if cmd_check(argparse.Namespace(**{**vars(args), "max_stale": 5.0, "max_entry": 10.0})) != 0:
        return 1
    _park_on_sigterm()
    bus, _ = open_bus(args.port, args.calibration)
    arm = Arm(bus, speed_deg_s=args.speed)
    try:
        print(f"  {YELLOW}energising{OFF} -- torque on, holding the current pose")
        arm.energise()
        entry, correction = entry_pose({n: arm.home[n] for n in JOINTS})
        if correction > 0.01:
            arm.glide(entry, f"entering the limit envelope ({correction:.2f} deg)")
        return int(body(arm))
    except BaseException as exc:  # noqa: BLE001 -- parking matters more than the type
        print(f"\n  {RED}{type(exc).__name__}{OFF}: freezing, then parking")
        raise
    finally:
        arm.park()
        bus.disconnect(disable_torque=True)


def cmd_hold(args: argparse.Namespace) -> int:
    """Energise and hold still. The e-stop rehearsal: nothing is commanded."""

    def body(arm: Arm) -> int:
        print(f"\n  {DIM}f = freeze   l = LIMP (it will fall)   q = park and quit{OFF}\n")
        deadline = time.time() + args.seconds
        for key in _keys():
            if key == "q" or time.time() > deadline:
                return 0
            if key == "f":
                arm.freeze()
                print("  frozen")
            if key == "l":
                arm.limp()
                print(f"  {RED}LIMP{OFF} -- torque off")
                return 0
            now = arm.present()
            drift = max(abs(now[n] - arm.home[n]) for n in JOINTS)
            print(f"\r  holding, worst drift from start {drift:5.2f} deg "
                  f"({deadline - time.time():4.0f}s left) ", end="", flush=True)
            time.sleep(0.1)
        return 0

    return _energised_session(args, body)


def cmd_jog(args: argparse.Namespace) -> int:
    """Move ONE joint by a small amount. Proves sign and scale before teleop."""
    def body(arm: Arm) -> int:
        start = arm.present()[args.joint]
        wanted = start + args.deg
        target = clamp_deg(args.joint, wanted)
        if abs(target - wanted) > 1e-9:
            print(f"  {YELLOW}clamped{OFF} {wanted:+.2f} -> {target:+.2f} deg (limit)")
        print(f"  {args.joint}: {start:+.2f} -> {target:+.2f} deg")
        arm.glide({args.joint: target}, "jogging")
        time.sleep(0.4)
        reached = arm.present()[args.joint]
        print(f"  reached {reached:+.2f} deg, error {reached - target:+.2f} deg")
        return 0

    return _energised_session(args, body)


def cmd_serve(args: argparse.Namespace) -> int:
    """Own the bus. Take targets over UDP, hold when they stop arriving.

    The fast half of the two-loop split: perception sends absolute targets at
    whatever rate it manages, this runs at `--rate` and interpolates toward the
    newest one. Nothing about the arm's motion is paced by the camera.
    """
    from robo_mimic.link import STALE_MS, Receiver, ReportSender

    def body(arm: Arm) -> int:
        try:
            receiver = Receiver(port=args.port_udp)
        except OSError as error:
            raise SystemExit(
                f"\n  cannot bind udp {args.port_udp}: {error}\n"
                f"  Another `arm.py serve` is almost certainly still running.\n"
                f"  Find it with:  pkill -INT -f 'arm.py serve'   (INT so it parks)\n"
            ) from error
        reporter = ReportSender(port=args.report_port)
        print(f"  listening on udp {args.port_udp}, reporting pose on {args.report_port}")
        print(f"  {args.rate:.0f} Hz control loop, {args.speed:.0f} deg/s ceiling")
        print(f"  {DIM}silence for {STALE_MS:.0f} ms freezes the arm. ctrl-c parks it.{OFF}\n")
        dt = 1.0 / args.rate
        step_ceiling = args.speed * dt  # degrees per tick, the hard rate limit

        # The setpoint we RAMP, distinct from where the arm actually is.
        #
        # Ramping from Present_Position instead looks equivalent and is not. A
        # servo under gravity needs a STANDING position error to hold at all --
        # measured 2.79 deg on the elbow with lerobot's P=16, I=0. Commanding
        # `present + 0.5` hands it 18% of the error it needs merely to stay put,
        # so a loaded joint sags while you believe you are raising it, and the
        # setpoint can never get ahead of actual. Exactly the phase-5
        # `advance()` bug (docs/measurements/phase-5-sim.md), reintroduced on
        # hardware, where it costs more.
        if args.start == "ready":
            # The arm cannot reach this by itself from limp and cannot hold it
            # without torque, so getting there is our job, and putting it back
            # is too -- `park` returns to the folded pose we found it in.
            arm.glide(READY, "moving to the READY pose")
        setpoint = {n: arm.present()[n] for n in JOINTS}

        last_seen = time.monotonic()
        holding = True
        seen_sequence = -1
        ticks = frozen = 0
        try:
            while True:
                tick = time.monotonic()
                target = receiver.drain()
                if target is not None and target.sequence != seen_sequence:
                    seen_sequence = target.sequence
                    last_seen = tick

                present = arm.present()
                reporter.send({n: present[n] for n in JOINTS},
                              present[GRIPPER] / 100.0)

                age_ms = (tick - last_seen) * 1000.0
                live = target is not None and age_ms <= STALE_MS and target.engaged
                if not live:
                    if not holding:
                        arm.freeze()
                        holding = True
                    # Re-seed, so resuming ramps from where it actually stopped
                    # rather than from a setpoint that kept running while frozen.
                    setpoint = {n: present[n] for n in JOINTS}
                    frozen += 1
                else:
                    holding = False
                    goal = {}
                    for name in JOINTS:
                        delta = target.joints[name] - setpoint[name]
                        capped = max(-step_ceiling, min(step_ceiling, delta))
                        setpoint[name] = clamp_deg(name, setpoint[name] + capped)
                        goal[name] = setpoint[name]
                    goal[GRIPPER] = min(100.0, max(0.0, target.gripper * 100.0))
                    arm.bus.sync_write("Goal_Position", goal)

                ticks += 1
                if ticks % int(args.rate) == 0:
                    state = "live" if live else "HOLD"
                    print(f"\r  {state}  rx {receiver.received:6d}  "
                          f"stale {age_ms:6.0f} ms  frozen {frozen * dt:5.1f} s ",
                          end="", flush=True)
                time.sleep(max(0.0, dt - (time.monotonic() - tick)))
        except KeyboardInterrupt:
            print("\n  interrupted")
            return 0
        finally:
            receiver.close()

    return _energised_session(args, body)


def cmd_align(args: argparse.Namespace) -> int:
    """Disarm a stale goal. Torque is never enabled, so nothing can move."""
    bus, _ = open_bus(args.port, args.calibration)
    try:
        torque = bus.sync_read("Torque_Enable", normalize=False)
        if any(torque.values()):
            print(f"  {RED}refusing{OFF}: the arm is ENERGISED. Aligning now would "
                  "command it to where it currently sags, which is a real move.")
            print(f"  {DIM}park or power down first, then run this{OFF}")
            return 1
        before = align(bus)
        after = stale_goal_deg(
            bus.sync_read("Present_Position", normalize=False),
            bus.sync_read("Goal_Position", normalize=False),
            RESOLUTION,
        )
        print(f"  stale goal {before:.2f} -> {after:.2f} deg. Nothing moved; "
              f"the next power-up is now a no-op.")
        return 0
    finally:
        bus.disconnect(disable_torque=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", default=DEFAULT_PORT, help="FOLLOWER port, never the leader")
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="read-only pre-flight. Touches no torque.")
    check.add_argument("--max-stale", type=float, default=5.0,
                       help="degrees of stale Goal_Position to tolerate (default 5)")
    check.add_argument("--max-entry", type=float, default=10.0,
                       help="degrees of unbidden power-up correction to tolerate (default 10)")
    check.set_defaults(func=cmd_check)

    hold = sub.add_parser("hold", help="energise and hold still. Rehearses the stop.")
    hold.add_argument("--seconds", type=float, default=20.0)
    hold.add_argument("--speed", type=float, default=15.0, help="deg/s ceiling")
    hold.set_defaults(func=cmd_hold)

    jog = sub.add_parser("jog", help="move ONE joint a little. Proves sign and scale.")
    jog.add_argument("--joint", required=True, choices=list(JOINTS))
    jog.add_argument("--deg", type=float, required=True)
    jog.add_argument("--speed", type=float, default=15.0, help="deg/s ceiling")
    jog.set_defaults(func=cmd_jog)

    aln = sub.add_parser("align", help="disarm a stale goal. Never energises.")
    aln.set_defaults(func=cmd_align)

    srv = sub.add_parser("serve", help="own the bus; take targets from teleop over UDP")
    srv.add_argument("--rate", type=float, default=50.0, help="control loop Hz")
    srv.add_argument("--speed", type=float, default=30.0, help="deg/s ceiling per joint")
    srv.add_argument("--port-udp", type=int, default=47101)
    srv.add_argument("--report-port", type=int, default=47102)
    srv.add_argument("--start", choices=("ready", "asis"), default="ready",
                     help="'ready' drives to the mid-range teleop pose first "
                          "(recommended); 'asis' begins wherever the arm sits")
    srv.set_defaults(func=cmd_serve)

    args = parser.parse_args()
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("\n  interrupted before anything was energised")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
