# Phase 5 — sim in the loop, measured

**Date**: 2026-09-17 · `src/robo_mimic/sim.py`, `scripts/run_sim.py` · 11 sim tests

The first phase where the whole pipeline runs end to end and an arm moves.

```
fixture landmarks → moved on a known path → HandLandmarks
  → hand_pose / image_position → Retargeter → project → SafetyLimiter
  → SimArm (500 Hz, interpolating) → measure
```

## The model

Fetched from `TheRobotStudio/SO-ARM100`, **hash-pinned**: 28 asset files,
**16.4 MB** of STL. Not vendored.

Verified: phi's vendored `so101_new_calib.xml` is **byte-identical** to upstream
(`d75253eb…`), so fetching and copying are equivalent. Fetching wins only on
self-containment. The pin exists because every kinematic constant in this repo
was measured from that exact file.

Loads: `njnt=6 nu=6 nv=6`, timestep `0.002 s` (500 Hz), `kp=998.22 kv=2.731`.

## Why a synthetic hand, not a recorded clip

A recorded clip has **no ground truth** — you can see whether the arm followed,
but not by how much it should have. The synthetic trajectory uses the committed
fixture's *real* MediaPipe landmarks, translated and scaled along a **known**
path, with noise at the measured level. Every error below is against something
exact.

`make replay CLIP=…` drives the identical pipeline from a real recording.

## THE GATE: does the two-loop split matter?

31 Hz perception into 500 Hz actuators with ζ = 0.250 and a 17.9 Hz resonance.
8 s trajectory, noise at the measured level:

| fast loop | settled p95 | **PEAK p95** | peak max |
|---|---|---|---|
| hold (31 Hz staircase) | 0.9750° | **2.7884°** | 3.6516° |
| **interpolate (500 Hz)** | **0.1544°** | **0.2089°** | **0.3608°** |

**Interpolating cuts peak tracking error 13.3×.**

Holding the slow loop's target between updates *is* a staircase, and a staircase
rings an underdamped arm. Ramping toward it does not. This is the architecture
the ECE 4560 students used — their terminal printed `Detect Freq: 31 Hz` against
`Arm Freq: 720–930 Hz` — and it is the cheapest improvement in the project.

> **Peak, not settled, is the number that matters.** The settled error at the
> end of each tick hides the ripple that lives between substeps: 0.9750° settled
> versus 2.7884° peak on the staircase. Reporting only the former would have
> made it look fine.

## Noise sensitivity, interpolating

| image noise (nu) | settled p95 | peak p95 | **tool p95** |
|---|---|---|---|
| 0.00000 (none) | 0.1497° | 0.1631° | **0.414 mm** |
| **0.00035** (measured, σ=1 level) | 0.1544° | 0.2089° | **0.442 mm** |
| 0.00100 | 0.7055° | 1.0879° | 3.200 mm |
| 0.00200 | 1.4314° | 2.4160° | 6.918 mm |

**At the measured camera-noise level, tool tracking is 0.442 mm p95.** That is
the honest answer to "how accurate will this be": **sub-millimetre in sim**, and
only 7% worse than the noise-free case, because the adaptive filter absorbs it.

Degradation is graceful but super-linear: 3× the noise gives 7× the error.

248 perception frames, 248 engaged, 0 held.

## Two bugs the tests found

### 1. `advance()` restarted its ramp every call

`set_target` shifted `_previous` **and** `advance` interpolated from it, so
calling `advance` repeatedly without a new target swept from the *old* target
each time. The command oscillated instead of settling.

**Caught by a convergence test**: the arm settled **9.2014° away** from a held
target after 1.92 s. Fixed by having `advance` consume `_previous`.

This did not affect the numbers above — `run_sim.py` calls `set_target` every
frame — but it would have broken any caller that stepped faster than it
retargeted.

### 2. A test that asserted something false

I asserted the jaw cannot move the tool at all. Two different decouplings, and
only one is exact:

| | |
|---|---|
| **kinematic** — `forward_kinematics` at gripper 0 vs 1.5 rad | **exactly 0.000e+00 m** |
| **dynamic** — sim, jaw fully open vs shut | **3.19e-06 m** |

Opening the jaw moves real mass, changing the gravity torque on the arm and
therefore its sag. Real physics, negligible in size, **but not zero**. The test
now checks the right thing and records why.

## Gate: PASSED

| requirement | result |
|---|---|
| pipeline runs end to end | ✅ 248 frames, perception → joints → moving arm |
| tracking-error table | ✅ above |
| rendered clip | ✅ `outputs/phase5-sim.mp4` |
| interpolation justified by measurement | ✅ **13.3×** |
| sim suite | ✅ 11 tests, `make check` still 195 with no camera or model |

## Still open

- The **velocity caps** (120 °/s, 2.5/s) remain **[UNVERIFIED]** on hardware.
- Sim has **no backlash**: the model declares a `backlash` class and never uses
  it (`njnt=6`). Real repeatability will be worse.
- Sim is **1.75× stronger** than our arm: the model is built for the 12 V
  servo (3.35 N·m) and ours is 7.4 V (1.91 N·m).
