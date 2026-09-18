# Phase 4 (filter half) — the safety layer, measured

**Date**: 2026-09-17 · module `src/mirror/safety.py` · 33 tests

## The contract

> For **any** input — `None`, NaN, ±inf, `1e9`, a malformed dict, a bad timestep —
> the emitted joint vector is finite, inside the **model's** limits, and within
> the velocity cap.

Stated as an invariant and fuzzed, not demonstrated by example.

## Why a filter is unavoidable

σ = 1 brightness level of camera noise → 0.226 mm lateral / 2.331 mm depth of
tool motion → the IK amplifies at ≈ **0.6 °/mm**. Joint tremor of a degree or
more is **intrinsic** to single-webcam tracking.

## Why it must be adaptive

| | |
|---|---|
| perception rate | 31.0 Hz (camera-bound) |
| its Nyquist | **15.5 Hz** |
| arm resonance | **17.9 Hz** (ζ = 0.250) |

**The resonance sits above the perception Nyquist.** We cannot observe it at
31 Hz — a 17.9 Hz disturbance aliases to 13.1 Hz. So the filter must cut well
below 15.5 Hz and assume the worst above it.

But cutting low costs lag:

| cutoff | first-order lag | attenuation @ 15.5 Hz | frames @ 31 Hz |
|---|---|---|---|
| 1 Hz | 159.2 ms | −23.8 dB | 4.93 |
| 2 Hz | 79.6 ms | −17.9 dB | 2.47 |
| 5 Hz | 31.8 ms | −10.3 dB | 0.99 |

A fixed cutoff must pick one point and live with it. **An adaptive one does
not**: filter hard when the hand is still (jitter visible, lag invisible), open
up when it moves (lag visible, jitter invisible).

```
cutoff(t) = fc_min + beta · |smoothed velocity|
```

## Tuning: the criterion matters more than the sweep

Realistic profile — two bursts separated by stillness, 1.5° RMS input noise:

| setting | tremor when STILL | error when MOVING |
|---|---|---|
| unfiltered | 1.5191° | 1.3620° |
| fc 0.5, β 0.02 | 0.4365° ✅ | **2.0995° ❌ worse than raw** |
| fc 0.5, β 0.05 | 0.5597° ✅ | 1.4719° ❌ worse than raw |
| **fc 0.5, β 0.10** | **0.6940° ✅ 2.2×** | **1.2079° ✅** |
| fc 0.5, β 0.20 | 0.8662° ✅ | 1.1017° ✅ |

Minimising one blended RMS picks **β = 0.02** — and that setting is *worse than
no filter at all* while the operator is moving. The blend is dominated by the
1147 still frames out of 1199.

**The right criterion is "beat the raw signal in BOTH regimes."** Only β ≥ 0.10
does. Chosen: `fc_min = 0.5 Hz, β = 0.10`.

## Filter and rate limiter are not the same job

A low-pass attenuates **small, fast** noise. A rate limiter caps **large, fast**
steps. Jitter passes straight through a rate limiter (already under the cap); a
tracking teleport passes a low-pass as a big smooth lunge.

## The fuzz gate

30 000 adversarial frames — `None`, NaN, ±inf, ±1e9, malformed dicts, wild
finite values, bad timesteps:

| | result |
|---|---|
| out-of-limit outputs | **0** |
| illegal jaw values | **0** |
| worst joint velocity | **120.0000 °/s** (cap 120.0) |
| worst jaw rate | **2.5000 /s** (cap 2.5) |
| frames the arm actually moved | 299 |

**At the caps, never over.** The last row matters: a limiter that never moves
is a brake, not a limiter.

Realistic stream (smooth motion + noise + 3% dropouts): **2917/3000 frames
moved, 83 held**, 0 escapes.

## Three bugs fuzzing found

### 1. A first-command exemption worth 3410 °/s

The first command was adopted directly, on the reasoning that filtering from an
arbitrary origin would drag the arm in from nowhere. **Fuzzing broke it in three
frames**: two dropped frames *report* the home posture while leaving the internal
state unset, so the next real command took the exemption and moved a joint
**109.9999° in one frame — 3410 °/s against a 120 °/s cap.**

Fixed by deleting the exemption. The limiter now always starts from a known
posture and rate-limits every frame including the first. Startup became a
controlled ramp, which is what was wanted anyway. On hardware: read
`Present_Position`, pass it to `reset()`.

### 2. A guard that froze the arm permanently

The discontinuity guard compared each input to the previous **output**. Because
the output lags behind — the rate limiter working as intended — a sustained fast
motion looks like a permanent discontinuity.

**Measured with that version: 30 000 frames, worst joint velocity `0.0000 °/s`.
Nothing ever moved.** Safe and completely useless.

Fixed by comparing input to the **previous input**. A one-frame glitch is a
spike there and is dropped; genuine fast motion is smooth and passes to the rate
limiter. The reference updates either way, so a target that really moved is
followed from the next frame — one frame of hold, not a refusal.

### 3. An unbounded adaptive cutoff

A `1e9` input makes the velocity estimate enormous, and the cutoff went to
**5.2e8 Hz** — a filter fully open to a rate that cannot represent anything near
it. Only the downstream rate limiter was preventing harm.

Fixed by capping the applied cutoff at `1/(2·dt)`, the Nyquist of the actual
sample interval. Nothing above it is observable, so claiming to pass it is
meaningless.

## Order of operations

```
reject non-finite → glitch guard → filter → rate limit → clamp
```

Clamping is **last**, which makes the limit guarantee unconditional. It also
cannot undo the rate limit: if the previous value is in range and the new one is
clamped into range, the clamp moves it *toward* the previous value, so
`|clamped − prev| ≤ |unclamped − prev|`.

## Still open

`max_velocity_deg_s = 120` and `max_gripper_rate_per_s = 2.5` are
**[UNVERIFIED]** against the physical STS3215. They must be checked on hardware
before Phase 7.

## Gate: PASSED

| requirement | result |
|---|---|
| no input escapes the limits | ✅ 0 in 30 000 adversarial frames |
| velocity cap respected | ✅ at the cap, never over |
| still usable | ✅ 2917/3000 frames moved on a realistic stream |
| filter beats raw in both regimes | ✅ 2.2× still, better moving |
| suite needs no camera | ✅ **195 passed**, 13 deselected |
