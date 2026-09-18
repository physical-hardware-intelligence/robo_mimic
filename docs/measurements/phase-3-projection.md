# Phase 3 — the projection, measured

**Date**: 2026-09-17 · module `src/robo_mimic/project.py` · 17 tests

The hole this fills: `retarget` emits a `Pose`, the IK wants `(position, pitch,
roll)`, and **nothing joined them up**. The IK was vendored, verified to
`0.00 pm`, and never called.

## The 5-DOF constraint is one equation

Measured over 3000 random in-limit poses — the tool frame's **x-axis**:

| claim | result |
|---|---|
| is the wrist/roll axis (unmoved by 37° of `wrist_roll`) | **7.85e-17** |
| is confined to the arm plane, always | **0.000000000** (min = median = max) |

```
â · n̂ = 0        â = tool x-axis,  n̂ = arm-plane normal
```

## Which rotation is lost

At fixed position the achievable rotations are a 2-parameter family. 1777
samples, 154 IK branch flips rejected:

| | min | median | max |
|---|---|---|---|
| \|pitch axis · n̂\| | **1.0000000** | 1.0000000 | 1.0000000 |
| \|roll axis · n̂\| | **0.0000000** | 0.0000000 | 0.0000000 |
| \|**missing** · n̂\| | **0.0000000** | 0.0000000 | 0.0000000 |
| \|**missing** · â\| | **0.0000000** | 0.0000000 | 0.0000000 |

⇒ the missing axis is `ĉ = n̂ × â`: **in the plane, perpendicular to the wrist.**

> **This corrects [ADR-001](../adr/001-five-dof-projection.md)**, which named the
> plane normal. Rotation about the normal *is* the pitch — fully achievable.

## Geometry, all verified against the FK

| formula | worst error |
|---|---|
| `n̂ = (sin pan, cos pan, 0)` | 1.25e-16 |
| `r̂ · n̂ = 0` | 1.67e-16 |
| `pitch = π − psi`, `psi = atan2(â·ẑ, â·r̂)` | **1.02e-13 deg** |

## Round trip

Feed back a pose the arm is already in — nothing should be given up:

| | result |
|---|---|
| status | **400/400 "exact"** |
| worst position error | **5.96e-16 m** |
| worst orientation error | **1.07e-12 deg** |

## The gate

Tilt a reachable pose about `ĉ` — the one forbidden direction — and the
projection must return exactly that tilt and nothing else:

| | result |
|---|---|
| \|residual axis · ĉ\| | **min 0.9937**, median 0.9994 |
| \|orientation error − tilt\| | **p99 0.148 deg** |
| `out_of_plane_deg` vs applied tilt | **exact to 1e-6 deg** (same-plane branch) |
| position error under a 60° tilt | **< 1e-9 m** — orientation loss never leaks into position |

**Not asserted as exact**, and that is deliberate: the roll correction is a
rotation about `â`, and rotations do not commute, so the total residual is a
composition rather than a single rotation about `ĉ`.

## Two independent losses, reported separately

| field | cause | can it be fixed? |
|---|---|---|
| `out_of_plane_deg` | five joints, six numbers | **no** — structural |
| `pitch_shift_deg` | joint limits | sometimes, by moving |
| `roll_clamp_deg` | `wrist_roll`'s ~40° dead sector | sometimes |

### Pitch gaps: the operating case is nothing like the random case

Random desired pitch → nearest reachable: **p50 49.5°, p90 124°, max 166°.**

Real operating case — hold the anchor rotation, nudge the position:

| nudge | out-of-plane | **pitch gap** | solved |
|---|---|---|---|
| 2 mm | 0.22° | **0.0°** | 100% |
| 10 mm | 1.30° | **0.0°** | 96% |
| 50 mm | 5.60° | **0.0°** | 84% |
| 100 mm | 12.07° | **0.0°** | 64% |

**The wanted pitch is simply reachable in normal use.** The out-of-plane cost is
the price of holding a world-fixed rotation while the plane rotates with the
pan: ≈ **1.3° per cm** of motion.

## Cost

| | |
|---|---|
| IK, failing early | 12.6 µs |
| FK | 111.6 µs |
| **`project()`** | **2.79 ms** |

Was **85.50 ms** before bounding the pitch sweep: the wrong pan branch swept the
full period (361 IK calls) because a given wrist axis is achievable on only one
branch. `MAX_PITCH_SHIFT_DEG = 30` caps it at 61 mostly-early-return calls.

**Budget**: `project` later fell to **0.30 ms** once the wasted pan branch was
found in Phase 6 — see [phase-6](phase-6-live.md). The 2.79 ms here is with both
branches swept.

## Bugs found, and how

| bug | found by | symptom |
|---|---|---|
| **roll sign inverted** | round-trip test | position exact to 6e-16 m, orientation up to **180° wrong**. `wrist_roll` turns about *minus* the tool x-axis: +7° of joint = −7.0000° about `â` |
| **roll silently reverted to 0** when out of limits | status-honesty test | ~150° error reported as `status="exact"`. 4 cases in 2982. Now **clamped** and flagged `roll_clamped` |
| **`in_limits` used phi's hand-typed table** | hypothesis totality test | a roll clamped to the model's own −157.2110° was rejected by phi's −157.2. `limits.py` is now the only source |
| **`DEGENERATE = 1e-9` too small** | hypothesis | normalising a 1e-9 vector amplifies error 10⁹×; an "in-plane" result came back 1.1e-12 out of plane. Raised to 1e-6 |

## Three wrong diagnoses I made along the way

Recorded because the pattern is the lesson.

1. **"A full pitch sweep does not fit in a frame."** It does: 360 × 12.6 µs = 4.5 ms of 33 ms.
2. **"Reachability is 0.8%, then 0.0%."** Both were my test: `tool_pitch` is *unwrapped* (−447°…+99°) while the IK only uses `cos`/`sin` of it, so it is effectively mod 2π. Over one period the answer is **19.9%**.
3. **"The gate fails because the branches are not antiparallel."** Partly, but the real cause was worse: I built the forbidden axis from `pan_candidates(p)[0]` when the pose's actual pan was the *other* branch, leaving `|â·n̂|` up to **0.14**. The cross product was not the forbidden axis at all.

**Every one came from reasoning where I could have measured.**

## Gate: PASSED

| requirement | result |
|---|---|
| residual confined to one axis | ✅ alignment ≥ 0.9937 over ~4000 poses |
| reported loss matches it | ✅ p99 0.148 deg |
| never fails | ✅ hypothesis, 400 arbitrary poses in and out of the workspace |
| in-limits always | ✅ against the model, not the vendored table |
| suite needs no camera | ✅ **162 passed**, 13 deselected |
