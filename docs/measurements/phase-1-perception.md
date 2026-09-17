# Phase 1 — perception, measured

**Date**: 2026-09-16
**Machine**: MacBook Air, Apple Silicon (arm64), macOS
**Stack**: `mediapipe 0.10.35` · `opencv 5.0.0` · `numpy 2.5.3` · Python 3.12.12
**Model**: `hand_landmarker.task` float16, 7.8 MB
  `sha256 fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1`
**Reference image**: 640×960, two hands

Everything below was measured, not estimated. Reproduce with `make fixtures`.

---

## 1. The version pin is load-bearing

`mediapipe 1.0.1` **aborts** on macOS arm64 inside its own graph:

```
F0000 graph_service.h:139] Check failed: service_ Service is unavailable.
  -[DrishtiMetalHelper initWithCalculatorContext:]
  mediapipe::api2::TensorsToDetectionsCalculator::Open()
```

`TensorsToDetectionsCalculator` is the **palm-detector** stage, and it initialises
the Metal helper unconditionally. **Passing `delegate=BaseOptions.Delegate.CPU`
does not avoid it** — tested, same abort.

| version | macOS arm64 wheel | runs |
|---|---|---|
| 1.0.1 | yes | ❌ aborts |
| 1.0.0 | yes | not tested |
| **0.10.35** | yes | ✅ |

Pinned to `mediapipe==0.10.35` in `pyproject.toml`.

---

## 2. Determinism

| test | deviation |
|---|---|
| 30 repeat `detect()` calls, one detector | **0.000e+00** |
| 5 freshly constructed detectors | **0.000e+00** |

**Bit-identical.** So the golden fixture asserts *exact equality*, not a tolerance.

⚠️ This holds for **one platform + one pinned version**. It is `[UNVERIFIED]`
across OS/arch — XNNPACK may take different kernel paths — which is why the
`golden` marker is excluded from CI and CI regenerates its own fixture.

---

## 3. The JPEG decoder changes the answer

This was the surprise, and it changed the fixture design.

Decoding the *same* progressive JPEG two ways:

| | |
|---|---|
| max pixel difference | **3 levels** |
| pixels differing | 51 354 / 1 843 200 (**2.79 %**) |
| mean difference | **0.0377 levels** |

That microscopic difference moves the landmarks:

| what changed | world-landmark shift |
|---|---|
| image **format** (SRGBA vs SRGB), identical pixels | **0.0000 mm** |
| JPEG **decoder** (MediaPipe vs OpenCV), identical format | **0.7370 mm** |

**The decoder is the hazard, not the format.**

### The fix: transcode to PNG once

PNG is lossless, so every decoder produces identical bytes:

| | |
|---|---|
| MediaPipe vs OpenCV PNG decode | **0 levels** |
| resulting landmark deviation | **0.000000 mm** |

The fixture is now reproducible **by construction** rather than by hoping
everyone links the same JPEG library.

---

## 4. Noise sensitivity — the perception noise floor

Gaussian noise added to the image, then re-detected:

| added noise (σ, levels) | world-landmark shift |
|---|---|
| 0.5 | **1.209 mm** |
| 1.0 | 2.610 mm |
| 2.0 | 2.636 mm |
| 4.0 | **11.221 mm** |

**This is the number that matters most for Phases 3–5.** A decent webcam in good
light runs σ ≈ 1–3 levels; dim light is far worse. So **landmark jitter of a few
millimetres to a centimetre is intrinsic**, not a bug to be fixed upstream.

Two consequences:

1. **The filter in `safety.py` is not polish.** It is handling a real, measured,
   irreducible noise source.
2. **It strengthens [ADR-002](../adr/002-incremental-position-clutch.md).** Noise
   in a *differential* signal partially cancels frame to frame; noise in an
   *absolute* position does not. The clutch design was chosen for the depth
   problem and turns out to help here too.

---

## 5. Latency

100 detections on 640×960, after a warm-up call:

| percentile | latency |
|---|---|
| p50 | **11.64 ms** |
| p90 | **11.76 ms** |
| p99 | 11.99 ms |

**≈ 85 FPS at p90.** The distribution is unusually tight (min 11.35, max 12.51),
and warm-up is negligible (first call 11.8 ms vs second 12.3 ms).

**Budget set at 35 ms (3×)** in `test_latency_is_within_budget` — loose enough
that a busy machine does not fail the build, tight enough that losing the CPU
delegate or changing resolution trips it.

> Headroom check: the arm accepts commands at ~30 Hz (33 ms/frame). Perception
> uses **11.8 of those 33 ms**. Perception is not the bottleneck.

---

## 6. Coordinate systems, verified against the docs

| claim (from the docs) | measured | verdict |
|---|---|---|
| normalized `z` origin is **the wrist** | wrist z = `-3.3e-07`, `+2.8e-07` | ✅ |
| world landmarks are **metric** | palm span 9.91 cm / 9.44 cm (adult palm ≈ 9–11 cm) | ✅ |
| world origin is the "approximate **geometric center**" | see below | ⚠️ needs care |

### What the world origin actually is

Distance of each candidate reference point from the world origin:

| candidate | hand 0 | hand 1 |
|---|---|---|
| **mean of the 4 MCP knuckles (5, 9, 13, 17)** | **0.44 cm** | **0.75 cm** |
| middle MCP (9) | 0.98 cm | 1.02 cm |
| bounding-box centre | 1.62 cm | 1.82 cm |
| mean of all 21 landmarks | 2.65 cm | 2.54 cm |
| wrist (0) | 9.34 cm | 8.66 cm |

**It is the palm centre, not the landmark centroid.** The mean of all 21 points
is 2.5–2.7 cm away — using it as "the origin" would introduce a systematic
centimetres-scale offset.

**This is good news for Phase 2**: the knuckles barely move relative to each
other, so this origin is stable under finger motion. A centroid of all 21 points
would wander every time a finger curled.

### The two outputs are not the same data rescaled

Ratio of world to image pair-distances, across all landmark pairs:

| hand | min | median | max | spread |
|---|---|---|---|---|
| 0 | 0.346 | 0.533 | 0.978 | **2.83×** |
| 1 | 0.289 | 0.686 | 1.077 | **3.72×** |

A pure rescale would give a spread of exactly **1.00×**. It does not, so the
world output is a **genuinely separate 3-D reconstruction**, not a scaled
projection of the image landmarks. **That is why orientation taken from `world`
is trustworthy** — it carries real depth structure about the hand's shape.

### But neither gives absolute position

Both origins sit **on the hand** — the wrist for image `z`, the palm centre for
world. Neither is the camera. **Absolute distance to the hand is simply absent**,
which is the entire basis of [ADR-002](../adr/002-incremental-position-clutch.md).

---

## 7. Model estimation noise between two hands

The reference image shows one person's two hands, so anatomy should match:

| | hand 0 | hand 1 | difference |
|---|---|---|---|
| palm span | 9.91 cm | 9.44 cm | **5 %** |
| hand length | 17.07 cm | 15.49 cm | **10 %** |

The two hands are not actually 10 % different sizes. **That spread is model
estimation error**, and it is another reason `pinch_ratio` divides by palm span
rather than thresholding metres.

---

## Gate: PASSED

| requirement | result |
|---|---|
| landmarks extracted, reproducible | ✅ bit-identical, PNG-pinned |
| per-frame latency table | ✅ p50 11.64 / p90 11.76 / p99 11.99 ms |
| fixtures frozen | ✅ `fixtures/landmarks_golden.npz`, 4.3 KB |
| pure suite runs with no camera and no mediapipe | ✅ 79 passed, 13 deselected |
