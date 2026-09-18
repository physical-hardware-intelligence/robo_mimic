# Phase 2 — hand frame, position proxy, clutch

**2026-09-16**, revised **2026-09-17** after live testing · same stack as
[phase-1](phase-1-perception.md)

## The frame

```
origin  palm centre = mean of the 4 MCP knuckles
y       toward the fingers (wrist -> middle knuckle)
z       out of the palm
x       y × z
```

Knuckles because they are the only rigid part. A frame built on fingertips
rotates when you curl a finger; the knuckle mean does not.

## Two things that would have broken it

### 1. The palm triangle is not square

| | `forward · across` | angle |
|---|---|---|
| hand 0 | **−0.2578** | 105° |
| hand 1 | **−0.4222** | 115° |

Assigning those directly as axes gives a **sheared** frame — not a rotation at
all. The cross product fixes it for free: `cross(a,b)` is perpendicular to both
*by definition*, so no Gram-Schmidt step is needed.

**Result**: `|RᵀR − I| = 1.1e-16`, `det = 1.000000000000`.

### 2. Left and right hands mirror

Index-knuckle → pinky-knuckle runs the **opposite way around** a left palm, so
`forward × across` points out of one palm and **into** the other.

| | |
|---|---|
| Left normal · Right normal, uncorrected | **−0.5687** |

Uncorrected, **swapping hands flips the robot.** This is anatomy, not a quirk of
one photo.

**Proof** — mirror a right hand into a left one. Under a reflection `M`, real
directions (fingers, palm normal) map through `M`; the third axis must gain an
extra sign, because a mirror reverses handedness. *That asymmetry is what makes
it a test and not a tautology.*

| axis | expected | error |
|---|---|---|
| x | `−M·x` | **0.000e+00** |
| y | `M·y` | **0.000e+00** |
| z | `M·z` | **0.000e+00** |
| z, **without** the fix | `M·z` | **1.861** ← wrong |

### Known-answer test

A flat right hand — fingers `+y`, palm facing `+z`, thumb side `−x` — is built
so the frame *must* be the identity: `max |R − I| = 0.000e+00`. Any sign or
cross-product order error fails immediately and names the axis.

## The position proxy

A camera gives a **bearing**, not a position. A lateral move `X` at distance `d`
shifts the image by `X/d`; apparent span `s` goes as `1/d`. Divide and `d`
cancels:

```
lateral = (image offset from the optical axis) / s        depth = 1/s
```

| check | result |
|---|---|
| move 0.2 of frame width at span 0.2 | **+1.000** palm-span |
| double the apparent span (hand nearer) | depth **halves**, exactly 2.00× |
| move **one apparent span**, any span in [0.05, 0.45] | **exactly 1.0** |

That last row is [ADR-002](../adr/002-incremental-position-clutch.md)'s
no-calibration promise made checkable: a big hand far away and a small hand
close up fill the same pixels and read the same number.

### ⚠ Correction (2026-09-17): measure from the optical axis

The proxy originally divided `cx` — measured from the **frame corner**. A
constant offset looks harmless when only differences are used, but it **does not
cancel when the span changes**, because the span divides it. Hand held dead
centre, moved only in depth:

| span | phantom lateral motion |
|---|---|
| ×1.2 | 0.589 spans |
| ×1.5 | 1.179 spans |
| ×2.0 | **1.768 spans ≈ 18 cm** |

Pure depth motion was inventing lateral motion. `(cx − 0.5)/s` is zero at the
axis for every span, so it does not — now **exactly 0.000e+00**.

### ⚠ Correction (2026-09-17): screen axes are not world axes

`image_position` returns **(screen-x, screen-y, depth)**; the arm's world frame
is **(reach, lateral, up)**. Adding them element-wise by index gave:

| hand | tool went |
|---|---|
| palm **right** | **+5.0 cm of REACH** (should be lateral) |
| palm **up** | **−3.8 cm of LATERAL** (should be up) |

Live, that read as *"the arm doesn't move left or right"* — because left/right
was being spent on reach, which runs out against the workspace almost at once.

Replaced with an explicit `axis_map` matrix. **Two of the tests covering this
were asserting the bug**: they checked `delta[0]` — the reach axis — because
they had been written to confirm what the code did rather than what it should
do. That is why 195 green tests missed something visible in ten seconds of use.

## The clutch

> **Superseded.** This phase put the clutch on a pinch with hysteresis
> (`pinch_on 0.35` / `pinch_off 0.50`).
> [ADR-003](../adr/003-clutch-on-a-key-gripper-on-the-pinch.md) moved it to a
> held key and gave the pinch to the gripper, which **deleted** the hysteresis
> band — a boolean cannot chatter. Current table:
> [phase-4](phase-4-safety.md) and `tests/test_retarget.py::TABLE`.

What survives from this phase: the **grace period** (200 ms) through tracking
dropouts, and the property that **engaging commands exactly where the arm
already is** — `0.0 m, 0.0°`, no jump.

## Gate: PASSED

| requirement | result |
|---|---|
| orthonormality under fuzz | ✅ 400 poses × both hands, `< 1e-12` |
| clutch state machine table-covered | ✅ all transitions |
| suite needs no camera | ✅ |
