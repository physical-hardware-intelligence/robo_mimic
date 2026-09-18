"""The hardware boundary: our joint angles <-> the servo bus's units.

Pure. Nothing here imports lerobot, opens a port, or moves anything, so all of
it is tested without an arm. `scripts/arm.py` is the impure half.

Three facts about the real SO-101 shape this module, all measured on our arm
2026-09-18 and recorded in docs/measurements/phase-7-hardware.md:

1. **lerobot reports the five body joints in degrees and the gripper in 0..100.**
   Our pipeline speaks degrees and an openness in 0..1, so the gripper needs a
   conversion and the body joints do not.

2. **In DEGREES mode lerobot does NOT clamp on write.** Its RANGE_0_100 and
   RANGE_M100_100 paths bound the value; the DEGREES path is a bare
   `int(val * 4095 / 360 + mid)` straight into `Goal_Position`. So for the five
   body joints `safety.py` is the ONLY thing between a bad number and a hard
   stop. In sim that clamp was a nicety. Here it is the guard rail.

3. **The zero is not where you assume.** lerobot's degree zero is the midpoint
   of the range recorded during calibration, not the URDF's kinematic zero. The
   two agree on our arm (checked by FK: the measured rest pose lands the tool
   14.5 cm out and 1.4 cm below the shoulder, which is where a folded SO-101
   actually is), but it is a property of this calibration, not a guarantee.
   `check_calibration` re-checks it rather than trusting it.
"""

from __future__ import annotations

from dataclasses import dataclass

from .kinematics.limits import JOINTS, LIMITS_DEG, clamp_deg

#: lerobot names the gripper alongside the five body joints and suffixes every
#: key with `.pos`. Ours is a separate scalar, so the two vocabularies differ.
GRIPPER = "gripper"

#: lerobot's gripper normalisation is RANGE_0_100 over the calibrated travel.
#: Ours is an openness in [0, 1]. Neither is degrees; this is a pure rescale.
GRIPPER_UNITS = 100.0

#: Below this the servos are browning out and positions stop meaning anything.
#: Our arm reads 11.8-12.0 V. The 7.4 V variant of the STS3215 exists, so this
#: is a floor for "something is wrong", not a claim about which variant we own.
MIN_VOLTS = 6.0

#: STS3215 shuts itself down around 70 C. Ours idles at 31-33 C.
MAX_TEMP_C = 55


#: Where to start a teleop session, and why not the pose the arm rests in.
#:
#: The arm's own gravity rest -- folded, jaw down -- is the obvious choice and
#: it is the wrong one. Measured against it (2026-09-18):
#:
#:   pose                     reach   height   manipulability   limit margin
#:   gravity rest, jaw down   14.4    -1.4 cm    1.51e-08         -4.0 deg
#:   READY (this)             26.3   +12.0 cm    3.07e-08        +36.8 deg
#:
#: Three disqualifications, not one. It sits **4 deg outside** our shoulder_lift
#: limit, so we cannot even command it. Its tool is **below the shoulder**, so
#: there is nowhere to reach down to -- and reaching down is most of what a
#: teleoperated arm does. And it has the **lowest manipulability** of anything
#: measured, because a folded arm is near the edge of its workspace where hand
#: motion buys little arm motion.
#:
#: READY is mid-range on every joint: 36.8 deg of limit margin, twice the
#: manipulability, and 12 cm of height to descend through. It is deliberately
#: NOT the most manipulable pose available (the old sim start scored 4.82e-08)
#: because that one holds the arm high and extended, which is a longer fall if
#: torque ever drops.
#:
#: The arm cannot reach this pose by itself from limp, and cannot hold it
#: without torque, so `arm.py serve --start ready` drives it there on the way
#: in and returns it to the folded rest on the way out.
READY: dict[str, float] = {
    "shoulder_pan": 0.0,
    "shoulder_lift": -60.0,
    "elbow_flex": 60.0,
    "wrist_flex": 30.0,
    "wrist_roll": 0.0,
}


#: Our arm's own gravity rest, clamped into limits. The one pose it holds with
#: NO torque, which is what makes it the only safe place to end a session.
#: Measured by letting it go limp and reading back: shoulder_lift sags to
#: -104.04, which is 4 deg outside our envelope, so we park 4 deg above it and
#: accept that cutting torque drops it those last 4 deg.
FOLDED: dict[str, float] = {
    "shoulder_pan": 0.0,
    "shoulder_lift": -100.0,
    "elbow_flex": 96.0,
    "wrist_flex": 77.0,
    "wrist_roll": 0.0,
}


@dataclass(frozen=True)
class JointReport:
    """One joint's answer to "can we safely command this?"."""

    joint: str
    present_deg: float
    limit_lo: float
    limit_hi: float

    @property
    def inside(self) -> bool:
        return self.limit_lo <= self.present_deg <= self.limit_hi

    @property
    def excursion_deg(self) -> float:
        """How far outside the limits it sits. Zero when inside."""
        if self.present_deg < self.limit_lo:
            return self.limit_lo - self.present_deg
        if self.present_deg > self.limit_hi:
            return self.present_deg - self.limit_hi
        return 0.0


def to_lerobot(joints: dict[str, float], gripper_openness: float) -> dict[str, float]:
    """Our units -> lerobot's `send_action` dict.

    Body joints pass through as degrees. The gripper rescales 0..1 -> 0..100.
    The openness is clamped here because lerobot's RANGE_0_100 path would clamp
    it anyway, and silently: better to be the one who decided.
    """
    openness = min(1.0, max(0.0, gripper_openness))
    action = {f"{name}.pos": float(joints[name]) for name in JOINTS}
    action[f"{GRIPPER}.pos"] = openness * GRIPPER_UNITS
    return action


def from_lerobot(observation: dict[str, float]) -> tuple[dict[str, float], float]:
    """lerobot's `get_observation` dict -> (body joints in degrees, openness)."""
    joints = {name: float(observation[f"{name}.pos"]) for name in JOINTS}
    openness = float(observation[f"{GRIPPER}.pos"]) / GRIPPER_UNITS
    return joints, openness


def report_pose(present_deg: dict[str, float]) -> list[JointReport]:
    """Where each joint sits relative to the limits we are willing to command."""
    return [
        JointReport(name, present_deg[name], *LIMITS_DEG[name])
        for name in JOINTS
        if name in present_deg
    ]


def entry_pose(present_deg: dict[str, float]) -> tuple[dict[str, float], float]:
    """The first pose we may command, and how far it is from where the arm is.

    The arm does not power up inside our envelope. Ours rests with
    `shoulder_lift` at -104.04 deg against a -100 deg limit, so the very first
    clamped command is a 4 deg move that nobody asked for. Making that explicit
    and measurable is the whole point of this function: the caller can refuse,
    or ramp into it, but it cannot be surprised by it.

    Returns the clamped pose and the largest per-joint correction in degrees.
    """
    clamped = {
        name: clamp_deg(name, value)
        for name, value in present_deg.items()
        if name in LIMITS_DEG
    }
    worst = max((abs(clamped[n] - present_deg[n]) for n in clamped), default=0.0)
    return clamped, worst


def check_calibration(
    calibration: dict[str, dict[str, int]], resolution: int = 4095
) -> list[tuple[str, float, float, bool]]:
    """Is every joint's calibrated travel at least as wide as what we command?

    lerobot's `range_min`/`range_max` are the tick extremes THIS arm reached
    during calibration, so they bound what it can physically do. Our limits come
    from the MuJoCo model's `ctrlrange`. If ours were the wider of the two we
    would be commanding into a mechanical stop, so this must hold for every
    joint before anything is energised.

    Returns `(joint, calibrated_span_deg, our_span_deg, ours_is_subset)`.
    """
    rows = []
    for name, entry in calibration.items():
        span_ticks = entry["range_max"] - entry["range_min"]
        calibrated = span_ticks * 360.0 / resolution
        if name in LIMITS_DEG:
            lo, hi = LIMITS_DEG[name]
            ours = hi - lo
        else:  # the gripper: lerobot clamps it itself, so any span is safe
            ours = 0.0
        rows.append((name, calibrated, ours, ours <= calibrated))
    return rows


def stale_goal_deg(
    present_ticks: dict[str, int], goal_ticks: dict[str, int], resolution: int = 4095
) -> float:
    """Worst gap between where the arm is and what its servos are still aiming at.

    Enabling torque makes every servo drive to whatever `Goal_Position` already
    holds, which is the last value written in some previous session. A large gap
    here means energising the arm will make it *snap*, and nothing in software
    gets a say. Ours reads 1.3 deg, which is why powering up is safe today. It
    is not safe by construction, so it is checked every time.
    """
    return max(
        abs(goal_ticks[name] - present_ticks[name]) * 360.0 / resolution for name in present_ticks
    )
