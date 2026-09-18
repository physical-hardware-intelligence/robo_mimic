# Phase 6 — live sim, measured

**2026-09-17** · `scripts/teleop.py`, `src/mirror/timing.py`

One command, three frame sources. `synthetic` skips capture and MediaPipe, so
the loop can be benchmarked with no camera permission.

## Real use beats synthetic benchmarks

Two sets of numbers, and only one of them is honest about the machine.

| stage | synthetic loop | **real session** |
|---|---|---|
| capture | 0.03 ms (stub) | **13.6 – 17.9 ms** |
| mediapipe | skipped | **11.7 ms** |
| retarget + project + safety + sim | 0.59 ms | **0.8 – 1.5 ms** |
| render | skipped in `--bench` | **~13 ms** |
| **total** | 0.59 ms | **40 – 64 ms → 13–20 fps** |

The synthetic figure (39.1 fps with rendering on) is a **ceiling, not a rate.**
Quote the real column.

## Why per-stage timing, not FPS

The loop ran at 113 fps and looked fine while `project` quietly burned
**8.47 ms** — more than MediaPipe, the largest stage in the pipeline.

| | |
|---|---|
| IK calls per frame | **62** (min = median = max) |
| a reachable pose needs | **1** |
| pan branch that won | first **0/120**, second **120/120** |

A wrist axis is reachable on **only one** pan branch; the other swept its full
61-step pitch bound for nothing, every frame.

**Fix, and it makes the result safer too:** try the branch nearest the arm's
current pan first (branch continuity was already required — switching branches
slams the arm), and stop on an exact solution.

**62 IK calls → 1. 8.47 ms → 0.30 ms.**

> Only *successful* IK calls are expensive, because `verify` runs FK inside each
> one. Unreachable targets are the cheap case: 122 calls in 1.9 ms, each
> returning early. I mis-blamed them first.

## MediaPipe: a benchmark I got wrong

I reported 3.87 ms in video mode and claimed a 3× saving. **That benchmark fed
the same frame repeatedly**, so tracking never lost lock and the palm detector
never re-ran.

On **181 distinct moving frames** from a real session:

| mode | mean | p95 | **hand found** |
|---|---|---|---|
| **video** | **12.17 ms** | 20.02 | **173/181 (96%)** |
| image | 23.98 ms | 31.37 | **67/181 (37%)** |

Video mode is right — but for **reliability**, not speed. Image mode loses the
hand on two thirds of moving frames. ~12 ms is the honest cost, and it matches
the 11.7 ms the live overlay reports.

## Rendering: throttling is the only lever

| | |
|---|---|
| `arm.frame(renderer)` | **23.36 ms** |
| `annotate()` | 0.65 ms |
| `side_by_side()` | 0.17 ms |
| `cv2.imshow` contention | **~11.9 ms** |

Resolution does **not** help — 640×480 costs 25.78 ms, 320×240 costs 22.87 ms.
A quarter of the pixels for 11% less time, so this is MuJoCo's per-call scene
setup and GL round-trip, not pixel fill.

| `--render-every` | synthetic fps | headroom at 31 Hz |
|---|---|---|
| 1 | 24.5 | **−2.31 ms ✗** |
| **2** (default) | 39.1 | **+12.36 ms ✓** |
| 3 | 46.0 | +14.96 ms |

Physics runs every frame regardless; only the display is throttled.

> phi's MuJoCo notes reached this from the other direction: redraw every 8th
> step, because a 60 Hz display cannot show 500 redraws/s.

## Gate: PASSED

| requirement | result |
|---|---|
| per-stage latency budget | ✅ real-session numbers above |
| runs live, one command | ✅ `make teleop` |
| verifiable without a camera | ✅ `make bench` |
| timing helper tested, not eyeballed | ✅ 13 tests |

**Open**: capture (~15 ms) is mostly the sensor wait and largely irreducible.
The camera path is verified by the operator, not by the agent — macOS grants
camera access per application.
