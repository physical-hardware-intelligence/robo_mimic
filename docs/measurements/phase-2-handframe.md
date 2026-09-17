# Phase 2 — hand frame + clutch, measured

**Date**: 2026-09-17 · same stack as [Phase 1](phase-1-perception.md)

## The frame

```
origin  palm centre = mean of the 4 MCP knuckles
y       toward the fingers (wrist -> middle knuckle)
z       out of the palm
x       y × z
```

## Two things that would have broken it

### 1. The palm triangle is not square

| | `forward · across` | angle |
|---|---|---|
| hand 0 | **−0.2578** | 105° |
| hand 1 | **−0.4222** | 115° |

Assigning `forward` and `across` directly as axes gives a **sheared** frame.
The cross product fixes it for free: `cross(a,b)` is perpendicular to both by
definition, so no Gram-Schmidt projection step is needed.

**Result**: `|RᵀR − I| = 1.1e-16` and `3.3e-16`, `det = 1.000000000000`.

### 2. Left and right hands mirror

Index-knuckle → pinky-knuckle runs the **opposite way around** a left palm than
a right one. So `forward × across` points out of one palm and **into** the other.

| | |
|---|---|
| Left normal · Right normal, uncorrected | **−0.5687** |

Uncorrected, swapping hands flips the robot.

**The fix**: `_PALM_NORMAL_SIGN = {"Right": −1, "Left": +1}`.

**The proof** — mirror a right hand into a left one. Under a reflection `M`,
real directions (fingers, palm normal) map through `M`; the third axis must gain
an extra sign, because a mirror reverses handedness.

| axis | expected | error |
|---|---|---|
| x | `−M·x` | **0.000e+00** |
| y | `M·y` | **0.000e+00** |
| z | `M·z` | **0.000e+00** |
| z, **without** the fix | `M·z` | **1.861** ← wrong |

## Known-answer test

A flat right hand, fingers `+y`, palm facing `+z`, thumb side `−x`, is built so
the frame must come out as **exactly the identity**.

```
max |R − I| = 0.000e+00
```

If any sign or cross-product order is wrong, this fails and names the axis.

## Position proxy

A camera gives a **bearing**, not a position. A lateral move `X` at distance `d`
shifts the image by `X/d`; apparent span `s` goes as `1/d`. Dividing cancels `d`:

```
lateral = (image offset) / s      depth = 1/s        -> units of palm-spans
```

| check | result |
|---|---|
| move 0.2 of frame width at span 0.2 | **+1.000** palm-span |
| double the apparent span (hand nearer) | depth **halves**, exactly 2.00× |
| move **one apparent span**, any span in [0.05, 0.45] | **exactly 1.0** |

That last row is ADR-002's no-calibration promise, made checkable: a big hand
far away and a small hand close up fill the same pixels and read the same, so
one `scale_m_per_span` suits every operator.

## The clutch

```
DISENGAGED --pinch--> ENGAGED --release--> DISENGAGED
                         └── hand lost > grace ──┘
```

All ten transitions are enumerated in `tests/test_retarget.py::TABLE`:

| start | input | → state | commands? |
|---|---|---|---|
| disengaged | open hand | disengaged | no |
| disengaged | pinch | **engaged** | yes |
| disengaged | in hysteresis band | disengaged | no |
| disengaged | no hand | disengaged | no |
| engaged | still pinched | engaged | yes |
| engaged | in hysteresis band | engaged | yes |
| engaged | released | **disengaged** | no |
| engaged | dropout 100 ms | engaged | yes (holds) |
| engaged | dropout 200 ms (at limit) | engaged | yes (holds) |
| engaged | dropout 201 ms | **disengaged** | no |

**Hysteresis** (`pinch_on 0.35`, `pinch_off 0.50`): a ratio oscillating inside
the band across 20 frames produces **zero** state changes. One threshold would
chatter and the arm would stutter.

**Grace period** (200 ms): MediaPipe drops frames. Disengaging on every blip is
unusable; never disengaging is unsafe.

### Properties, fuzzed

| claim | result |
|---|---|
| a disengaged clutch never commands motion | holds over 200 random sequences |
| engaging commands exactly where the arm already is | **0.0 m, 0.0°** — no jump |
| one drag cannot move the tool further than `5 × scale` | holds (screen x ∈ [0,1], span 0.2) |
| re-engaging rebases on the arm's current pose | **0.0 m** offset |

## Gate: PASSED

| requirement | result |
|---|---|
| orthonormality under fuzz | ✅ 400 poses × both hands, `< 1e-12` |
| clutch state machine table-covered | ✅ all 10 transitions |
| suite still needs no camera | ✅ **120 passed**, 13 deselected |
