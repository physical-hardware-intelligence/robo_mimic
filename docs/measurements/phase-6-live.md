# Phase 6 — live sim, measured

**Date**: 2026-09-17 · `scripts/teleop.py`, `src/mirror/timing.py` · 208 pure tests

```
capture → mediapipe → retarget → project → safety → sim.set_target
        → sim.advance (16 × 2 ms) → render
```

One command, three frame sources. `synthetic` bypasses capture and MediaPipe
entirely so the loop can be benchmarked on a machine with **no camera
permission** — which is exactly the situation this was built in.

## Why per-stage timing, not FPS

The loop ran at **113 fps and looked fine** while `project` quietly burned
**8.47 ms** — more than MediaPipe, the largest stage in the pipeline. No
aggregate number would have shown that. The per-stage table showed it
immediately.

### What `project` was doing

| | |
|---|---|
| IK calls per frame | **62** (min = median = max) |
| a reachable pose needs | **1** |
| pan branch that won | first candidate **0/120**, second **120/120** |

A given wrist axis is achievable on **only one** pan branch. The other was
sweeping its full 61-step pitch bound to no purpose, every frame.

**Two fixes, both of which also make the result safer:**

1. Try the branch **nearest the arm's current pan** first. Branch continuity was
   already a requirement — switching branches slams the arm — so this is not
   merely an optimisation.
2. Stop as soon as a branch yields an **exact** solution. Nothing on the other
   branch can beat exact, and the only remaining tiebreaker was joint travel,
   which fix 1 already minimises.

**62 IK calls → 1. `project`: 8.47 ms → 0.30 ms, a 28× cut.**

## MediaPipe is 3× cheaper than Phase 1 said

| mode | cost |
|---|---|
| IMAGE (Phase 1) | 11.8 ms |
| **VIDEO** | **3.87 ms** |

Video mode reuses the previous frame's hand box and **skips the palm detector**
while tracking holds. That is the two-stage design from Phase 1 paying off, and
it is why `teleop.py` uses video mode. The Phase 1 figure was not wrong, but it
was the wrong mode to quote for a live loop.

## Rendering is the bottleneck, and resolution does not help

Attributed precisely:

| | |
|---|---|
| `arm.frame(renderer)` | **23.36 ms** |
| `annotate()` | 0.65 ms |
| `side_by_side()` | 0.17 ms |
| `cv2.imshow` contention (window open vs not) | **~11.90 ms** |

| render size | ms/frame |
|---|---|
| 640×480 | 25.78 |
| 480×360 | 24.69 |
| 320×240 | **22.87** |
| 240×180 | 24.11 |

**Roughly flat** — ¼ the pixels for 11% less time. This is MuJoCo's per-call
scene setup and GL round-trip, not pixel fill, so **throttling the redraw is the
only lever that works.**

## `--render-every`, measured

| N | sustained | total | headroom at 31 Hz |
|---|---|---|---|
| 1 | 24.5 fps | 34.56 ms | **−2.31 ms ✗ does not fit** |
| **2** (default) | **39.1 fps** | 19.90 ms | **+12.36 ms ✓** |
| 3 | 46.0 fps | 17.30 ms | +14.96 ms |

Default **2**: physics runs every frame, the sim view refreshes at ~15 Hz, and
the loop keeps up with a 30 fps camera instead of falling behind and
accumulating lag.

> phi's own MuJoCo notes reached the same conclusion from the other direction:
> redraw every 8th step, because a 60 Hz display cannot show 500 redraws/s.

## The budget, after the fixes

| stage | mean |
|---|---|
| capture (camera) | ~1.00 ms |
| mediapipe (video mode) | 3.87 ms |
| retarget + project + safety + sim | **0.56 ms** |
| render, every 2nd frame | ~9.0 ms |
| **TOTAL** | **~14.4 ms of a 32.26 ms frame** |

**The loop is camera-bound, not compute-bound.**

## Gate: PASSED

| requirement | result |
|---|---|
| per-stage latency budget | ✅ above |
| sustained FPS | ✅ 39.1 fps at the default throttle |
| runs live, one command | ✅ `make teleop` |
| verifiable without a camera | ✅ `make bench` |
| timing helper tested, not eyeballed | ✅ 13 tests in `test_timing.py` |

## Still open

- The **camera path is [UNVERIFIED]** end to end by the agent: macOS grants
  camera access per application and refused this process. Everything downstream
  of the landmarks has 208 tests behind it, and `--source synthetic` exercises
  the identical loop.
- Velocity caps remain **[UNVERIFIED]** on hardware.
