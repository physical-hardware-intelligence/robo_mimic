"""The last thing between a noisy camera and a real servo. Pure, and total.

Nothing here is polish. Every stage exists because something measured demands
it, and the module's contract is a hard guarantee rather than a best effort:

    for ANY input -- NaN, empty, a hand teleporting across the frame, a
    commanded pose that flips IK branches -- the emitted joint vector is
    finite, inside the model's own limits, and within the velocity cap.

WHY A FILTER IS NEEDED AT ALL
-----------------------------
Measured: sigma = 1 brightness level of camera noise moves the landmarks enough
to shift the tool 0.226 mm laterally and 2.331 mm in depth, and the IK amplifies
tool error into joint error at roughly 0.6 deg/mm. So joint tremor of a degree
or more is INTRINSIC to single-webcam tracking. It is not a bug upstream and it
will not go away.

WHY THE FILTER MUST BE ADAPTIVE
-------------------------------
    perception rate   31.0 Hz  (camera-bound)
    its Nyquist       15.5 Hz
    arm resonance     17.9 Hz  (zeta = 0.250, 24.4 percent overshoot on a step)

The resonance sits ABOVE the perception Nyquist. We cannot even observe it at 31
Hz -- a 17.9 Hz disturbance aliases to 13.1 Hz in the sampled signal -- so the
filter has to cut well below 15.5 Hz and simply assume the worst above it.

But cutting that low costs lag, and lag is what makes teleoperation feel dead:

    cutoff    first-order lag    attenuation at 15.5 Hz    frames at 31 Hz
    1 Hz          159.2 ms             -23.8 dB                 4.93
    2 Hz           79.6 ms             -17.9 dB                 2.47
    5 Hz           31.8 ms             -10.3 dB                 0.99

A fixed cutoff must pick one point on that curve and live with it. An ADAPTIVE
cutoff does not: it filters hard when the hand is nearly still -- which is when
jitter is visible and lag is not -- and opens up when the hand moves, which is
when lag is visible and jitter is not. That is the 1-euro filter (Casiez,
Roussel, Vogel), designed for exactly this problem.

    cutoff(t) = fc_min + beta * |smoothed velocity|

WHY A RATE LIMIT TOO -- IT IS NOT THE SAME JOB
----------------------------------------------
A low-pass attenuates SMALL, FAST noise. A rate limiter caps LARGE, FAST steps.
Neither substitutes for the other: jitter passes straight through a rate limiter
(it is already under the cap), and a tracking glitch that teleports the hand
passes a low-pass as a big smooth lunge. They target different failure modes.

WHY A DISCONTINUITY GUARD, AND WHAT IT MAY BE MEASURED AGAINST
--------------------------------------------------------------
Transient garbage -- one frame of a flipped IK branch, a landmark glitch -- must
not be pursued at all. The rate limiter already bounds how FAST the arm chases
any target, so the guard's job is narrower: drop the frame entirely.

The subtlety is what "a jump" is measured against. Comparing the input to the
previous OUTPUT is wrong, and fuzzing showed why: because the output lags behind
(that is the rate limiter working), a sustained fast motion looks like a
permanent discontinuity and the arm freezes forever. Measured: with the guard
against the output, 30000 adversarial frames produced a worst joint velocity of
0.0000 deg/s -- nothing moved, ever.

So the guard compares each input to the PREVIOUS INPUT. A one-frame glitch is a
spike in that signal and is dropped; a genuine fast move is smooth in it and
passes through to the rate limiter. The reference is updated either way, so a
target that really has moved is followed from the next frame -- one frame of
hold, not a permanent refusal.

Branch switches that PERSIST are not this module's job: `project` already
prefers the branch nearest the current posture. All the rate limiter can promise
is that any pursuit happens at or below the velocity cap.

THERE IS NO FIRST-COMMAND EXEMPTION
-----------------------------------
An earlier version adopted the first command directly, on the reasoning that
filtering from an arbitrary origin would drag the arm in from nowhere. Fuzzing
found the hole in three frames: a couple of dropped frames report the home
posture while leaving the internal state unset, so the next real command took
the exemption and moved a joint 109.9999 deg in one frame -- 3410 deg/s against
a 120 deg/s cap.

So the limiter now always starts from a KNOWN posture and rate-limits every
frame from it, including the first. Startup becomes a controlled ramp instead of
a jump, which is what was wanted anyway. On hardware: read `Present_Position`,
pass it to `reset()`, and the first command is then measured from where the arm
actually is.

ORDER OF OPERATIONS, AND WHY CLAMPING IS LAST
---------------------------------------------
    reject non-finite -> discontinuity guard -> filter -> rate limit -> clamp

Clamping last is what makes the limit guarantee unconditional. It also cannot
undo the rate limit: if the previous value is in range and the new one is
clamped into range, the clamp moves it TOWARD the previous value, so
|clamped - previous| <= |unclamped - previous|. The velocity cap survives.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .kinematics.limits import GRIPPER_RAD, JOINTS, LIMITS_DEG, clamp_deg

#: Cutoff used to smooth the velocity estimate that drives the adaptive cutoff.
#: The standard 1-euro choice: low enough that noise in the derivative does not
#: itself widen the passband.
DERIVATIVE_CUTOFF_HZ = 1.0


def _alpha(cutoff_hz: float, dt_s: float) -> float:
    """Exponential-smoothing weight for a first-order low-pass.

    `tau = 1 / (2 pi fc)` and `alpha = 1 / (1 + tau/dt)`. Expressed in Hz and
    seconds rather than as a bare "smoothing factor" so the cutoff is a physical
    quantity that can be compared against the 15.5 Hz Nyquist above.
    """
    tau = 1.0 / (2.0 * np.pi * max(cutoff_hz, 1e-6))
    return float(dt_s / (tau + dt_s))


class OneEuroFilter:
    """Adaptive first-order low-pass: heavy when still, light when moving.

    Two parameters, both physical. `min_cutoff_hz` sets how hard a stationary
    signal is filtered; `beta` sets how quickly the cutoff opens with speed.
    """

    __slots__ = ("_beta", "_min_cutoff", "_prev", "_velocity")

    def __init__(self, min_cutoff_hz: float, beta: float) -> None:
        if min_cutoff_hz <= 0.0:
            raise ValueError("min_cutoff_hz must be positive")
        if beta < 0.0:
            raise ValueError("beta must be non-negative")
        self._min_cutoff = min_cutoff_hz
        self._beta = beta
        self._prev: float | None = None
        self._velocity = 0.0

    def reset(self) -> None:
        self._prev = None
        self._velocity = 0.0

    def __call__(self, value: float, dt_s: float) -> float:
        if self._prev is None:
            self._prev = value
            return value
        raw_velocity = (value - self._prev) / dt_s
        self._velocity += _alpha(DERIVATIVE_CUTOFF_HZ, dt_s) * (
            raw_velocity - self._velocity
        )
        self._prev += _alpha(self._cutoff(dt_s), dt_s) * (value - self._prev)
        return self._prev

    def _cutoff(self, dt_s: float) -> float:
        """Adaptive cutoff, capped at the Nyquist of the actual sample interval.

        The cap is not cosmetic. A 1e9 input makes the velocity estimate
        enormous and, uncapped, drove the cutoff to 5.2e8 Hz -- a filter fully
        open to a sample rate that cannot represent anything near it. Nothing
        above `1 / (2 dt)` is observable, so claiming to pass it is meaningless.
        Only the rate limiter downstream was preventing harm; now the filter
        does not need rescuing.
        """
        nyquist = 0.5 / max(dt_s, 1e-9)
        return min(self._min_cutoff + self._beta * abs(self._velocity), nyquist)

    def cutoff_at(self, dt_s: float) -> float:
        """The cutoff this filter would use right now, Hz. For the readout."""
        return self._cutoff(dt_s)

    @property
    def cutoff_hz(self) -> float:
        """Uncapped adaptive cutoff, Hz. Use `cutoff_at` for what is applied."""
        return self._min_cutoff + self._beta * abs(self._velocity)


@dataclass(frozen=True, slots=True)
class SafetyConfig:
    """Limits and filter settings.

    The velocity caps are [UNVERIFIED] against the physical servo: they are
    chosen to be brisk but not violent, and must be checked on hardware before
    Phase 7. Everything else is derived from measurements in this repo.
    """

    #: Per-joint velocity cap, degrees per second. At 31 Hz that is ~3.9 deg
    #: per frame. [UNVERIFIED] on the real STS3215.
    max_velocity_deg_s: float = 120.0
    #: Jaw rate cap, openness units per second. Slower than the arm on purpose:
    #: a jaw slamming shut is the one motion that can crush something.
    max_gripper_rate_per_s: float = 2.5
    #: Cutoff when the signal is stationary, Hz. Well under the 15.5 Hz
    #: perception Nyquist, and low because `beta` reopens it the moment the hand
    #: moves -- so the lag is paid only while still, where it is invisible.
    min_cutoff_hz: float = 0.5
    #: How fast the cutoff opens with speed, Hz per (deg/s).
    #:
    #: TUNED, not guessed. On a realistic profile (two bursts separated by
    #: stillness, 1.5 deg RMS input noise) against the criterion "beat the
    #: unfiltered signal in BOTH regimes" -- a stronger test than minimising one
    #: blended RMS, which is dominated by the 1147 still frames out of 1199:
    #:
    #:                     tremor when STILL    error when MOVING
    #:   unfiltered              1.5191 deg           1.3620 deg
    #:   fc 0.5, beta 0.02       0.4365 deg           2.0995 deg  <- WORSE moving
    #:   fc 0.5, beta 0.05       0.5597 deg           1.4719 deg  <- still worse
    #:   fc 0.5, beta 0.10       0.6940 deg           1.2079 deg  <- both better
    #:   fc 0.5, beta 0.20       0.8662 deg           1.1017 deg
    #:
    #: Below 0.10 the lag error while moving exceeds the noise it removes, so the
    #: filter is a net loss exactly when the operator is trying to do something.
    beta: float = 0.10
    #: A single-frame change in the INPUT beyond this is treated as a glitch and
    #: the frame is dropped. Measured against the previous input, not the
    #: previous output -- see the module docstring for why that distinction is
    #: the difference between a working guard and a permanently frozen arm.
    #:
    #: 30 deg is generous: a fast 10 cm/s hand at 31 Hz moves 3.2 mm per frame,
    #: which the IK's ~0.6 deg/mm turns into roughly 2 deg of joint change.
    max_jump_deg: float = 30.0

    def __post_init__(self) -> None:
        if self.max_velocity_deg_s <= 0.0 or self.max_gripper_rate_per_s <= 0.0:
            raise ValueError("rate caps must be positive")
        if self.min_cutoff_hz <= 0.0:
            raise ValueError("min_cutoff_hz must be positive")
        if self.beta < 0.0:
            raise ValueError("beta must be non-negative")
        if self.max_jump_deg <= 0.0:
            raise ValueError("max_jump_deg must be positive")


@dataclass(frozen=True, slots=True)
class SafeCommand:
    """What will actually be sent to the servos.

    `joints` and `gripper` are ALWAYS present and always legal. There is no
    None case: when there is nothing new to act on, this holds the last safe
    value, because "hold position" is a command and silence is not.
    """

    joints: dict[str, float]
    gripper: float
    #: True when this repeats the previous output rather than following input.
    held: bool
    reason: str
    #: Joints that hit a position limit this frame.
    clamped: tuple[str, ...] = ()
    #: Joints that hit the velocity cap this frame.
    rate_limited: tuple[str, ...] = ()
    #: The adaptive cutoff currently in use, Hz. For the readout.
    cutoff_hz: float = 0.0


@dataclass
class _State:
    """Always fully initialised. `joints` is never None -- see the module note
    on why the first-command exemption was removed."""

    filters: dict[str, OneEuroFilter]
    gripper_filter: OneEuroFilter
    joints: dict[str, float]
    gripper: float
    #: The previous INPUT, which is what the discontinuity guard measures
    #: against. Distinct from `joints`, the previous OUTPUT.
    last_input: dict[str, float] | None = None


class SafetyLimiter:
    """Filter, rate limit, clamp. The guarantee lives here.

    Stateful: filtering and rate limiting are both defined against history.
    Feed it frames in order and give it the real elapsed time -- `dt_s` is used
    as a physical quantity, so passing a nominal value when the true interval
    differs will mis-tune the filter.
    """

    def __init__(self, config: SafetyConfig | None = None) -> None:
        self.config = config or SafetyConfig()
        self.reset()

    @property
    def last(self) -> dict[str, float]:
        """The most recent emitted joint vector."""
        return dict(self._state.joints)

    @property
    def last_gripper(self) -> float:
        """The most recent emitted jaw opening, in [0, 1]."""
        return self._state.gripper

    def reset(self, joints: dict[str, float] | None = None, gripper: float = 0.0) -> None:
        """Re-seed the starting posture and drop filter history.

        Pass the arm's ACTUAL joint angles -- on hardware, `Present_Position` --
        so the next command is rate-limited from where the arm really is rather
        than from an assumed home. Defaults to all zeros.
        """
        start = (
            {j: clamp_deg(j, float(joints[j])) for j in JOINTS}
            if joints is not None
            else dict.fromkeys(JOINTS, 0.0)
        )
        jaw = float(np.clip(gripper, 0.0, 1.0))
        filters = {}
        for joint in JOINTS:
            filters[joint] = OneEuroFilter(self.config.min_cutoff_hz, self.config.beta)
            filters[joint](start[joint], 1.0 / 31.0)
        jaw_filter = OneEuroFilter(self.config.min_cutoff_hz, self.config.beta)
        jaw_filter(jaw, 1.0 / 31.0)
        self._state = _State(
            filters=filters,
            gripper_filter=jaw_filter,
            joints=start,
            gripper=jaw,
            last_input=None,
        )

    def step(
        self,
        joints: dict[str, float] | None,
        gripper: float | None,
        dt_s: float,
    ) -> SafeCommand:
        """One frame in, one legal command out. Never raises, never returns None."""
        if not np.isfinite(dt_s) or dt_s <= 0.0:
            return self._hold("bad dt")

        if joints is None or gripper is None:
            return self._hold("no command")

        try:
            wanted = {j: float(joints[j]) for j in JOINTS}
            wanted_gripper = float(gripper)
        except (KeyError, TypeError, ValueError):
            return self._hold("malformed command")

        if not all(np.isfinite(v) for v in wanted.values()) or not np.isfinite(
            wanted_gripper
        ):
            # A NaN reaching a servo is a moving robot with no idea where it is.
            return self._hold("non-finite command")

        previous_input = self._state.last_input
        self._state.last_input = wanted
        if previous_input is not None:
            jump = max(abs(wanted[j] - previous_input[j]) for j in JOINTS)
            if jump > self.config.max_jump_deg:
                # Dropped, but the reference above is already updated, so a
                # target that has genuinely moved is followed next frame.
                return self._hold(f"glitch {jump:.1f} deg")

        return self._follow(wanted, wanted_gripper, dt_s)

    # --- internals ---------------------------------------------------------
    def _follow(
        self, wanted: dict[str, float], wanted_gripper: float, dt_s: float
    ) -> SafeCommand:
        state = self._state
        step_cap = self.config.max_velocity_deg_s * dt_s
        clamped: list[str] = []
        rate_limited: list[str] = []
        out: dict[str, float] = {}

        for joint in JOINTS:
            smoothed = state.filters[joint](wanted[joint], dt_s)
            previous = state.joints[joint]

            limited = smoothed
            if abs(smoothed - previous) > step_cap:
                limited = previous + np.sign(smoothed - previous) * step_cap
                rate_limited.append(joint)

            final = clamp_deg(joint, float(limited))
            if final != limited:
                clamped.append(joint)
            out[joint] = final

        jaw = state.gripper_filter(float(np.clip(wanted_gripper, 0.0, 1.0)), dt_s)
        jaw_cap = self.config.max_gripper_rate_per_s * dt_s
        if abs(jaw - state.gripper) > jaw_cap:
            jaw = state.gripper + np.sign(jaw - state.gripper) * jaw_cap
            rate_limited.append("gripper")
        jaw = float(np.clip(jaw, 0.0, 1.0))

        state.joints = out
        state.gripper = jaw
        return SafeCommand(
            joints=dict(out),
            gripper=jaw,
            held=False,
            reason="tracking",
            clamped=tuple(clamped),
            rate_limited=tuple(rate_limited),
            cutoff_hz=state.filters[JOINTS[0]].cutoff_at(dt_s),
        )

    def _hold(self, reason: str) -> SafeCommand:
        """Repeat the last legal output.

        Holding, not zeroing. Commanding the home posture on a dropped frame
        would make every glitch a full-speed retreat across the workspace.
        """
        state = self._state
        joints = dict(state.joints)
        return SafeCommand(
            joints={j: clamp_deg(j, joints[j]) for j in JOINTS},
            gripper=float(np.clip(state.gripper, 0.0, 1.0)),
            held=True,
            reason=reason,
        )


def gripper_command_rad(openness: float) -> float:
    """Jaw opening -> servo radians, clamped to the model's own `ctrlrange`."""
    lo, hi = GRIPPER_RAD
    return float(np.clip(lo + np.clip(openness, 0.0, 1.0) * (hi - lo), lo, hi))


def velocity_of(
    current: dict[str, float], previous: dict[str, float], dt_s: float
) -> dict[str, float]:
    """Per-joint velocity in degrees per second. For tests and the readout."""
    return {j: (current[j] - previous[j]) / dt_s for j in JOINTS}


def all_in_limits(joints: dict[str, float]) -> bool:
    """The hard invariant, as a function so tests and callers agree on it."""
    return all(
        LIMITS_DEG[j][0] <= joints[j] <= LIMITS_DEG[j][1] and np.isfinite(joints[j])
        for j in JOINTS
    )


__all__ = [
    "DERIVATIVE_CUTOFF_HZ",
    "OneEuroFilter",
    "SafeCommand",
    "SafetyConfig",
    "SafetyLimiter",
    "all_in_limits",
    "gripper_command_rad",
    "velocity_of",
]
