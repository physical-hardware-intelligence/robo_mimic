# Phase 1 — perception, measured

**2026-09-16** · `mediapipe 0.10.35` · `opencv 5.0.0` · `numpy 2.5.3` · Python 3.12.12
· MacBook Air, arm64 · model `hand_landmarker.task` float16, 7.8 MB,
`sha256 fbc2a300…`

Reproduce: `make fixtures`.

## 1. The version pin is load-bearing

`mediapipe 1.0.1` **aborts** on macOS arm64 inside its own graph:

```
F0000 graph_service.h:139] Check failed: service_ Service is unavailable.
  -[DrishtiMetalHelper initWithCalculatorContext:]
  mediapipe::api2::TensorsToDetectionsCalculator::Open()
```

That calculator is the **palm detector**, and it initialises the Metal helper
unconditionally — **`delegate=CPU` does not avoid it** (tested). `0.10.35` runs
clean.

## 2. Determinism

| test | deviation |
|---|---|
| 30 repeat `detect()` calls, one detector | **0.000e+00** |
| 5 freshly constructed detectors | **0.000e+00** |

Bit-identical, so goldens assert *exact* equality. Holds for **one platform +
one pinned version** only; `[UNVERIFIED]` across OS/arch, which is why the
`golden` marker is excluded from CI.

## 3. The JPEG decoder changes the answer

This redesigned the fixtures.

Decoding the *same* progressive JPEG two ways:

| | |
|---|---|
| max pixel difference | 3 levels |
| pixels differing | 2.79 % |
| mean difference | **0.0377 levels** |

| what changed | world-landmark shift |
|---|---|
| image **format** (SRGBA vs SRGB), same pixels | **0.0000 mm** |
| JPEG **decoder** (MediaPipe vs OpenCV), same format | **0.7370 mm** |

**The decoder is the hazard, not the format.**

**Fix: transcode once to PNG.** Lossless, so every decoder agrees —
`0 levels` difference, `0.000000 mm` landmark deviation. The fixture is
reproducible by construction, not by hoping everyone links the same libjpeg.

## 4. The noise floor

Gaussian noise added to the image, then re-detected:

| σ (levels) | world-landmark shift |
|---|---|
| 0.5 | **1.209 mm** |
| 1.0 | 2.610 mm |
| 2.0 | 2.636 mm |
| 4.0 | **11.221 mm** |

A decent webcam in good light is σ ≈ 1–3. **Millimetre-to-centimetre jitter is
intrinsic**, not a bug to fix upstream.

Two consequences: the filter in `safety.py` is a safety component, not polish;
and noise in a *differential* signal partly cancels where an absolute one does
not — [ADR-002](../adr/002-incremental-position-clutch.md) turns out to be right
for a second, independent reason.

## 5. Latency

100 detections on 640×960, IMAGE mode, after warm-up:

| p50 | p90 | p99 |
|---|---|---|
| 11.64 ms | **11.76 ms** | 11.99 ms |

Unusually tight (min 11.35, max 12.51); warm-up negligible.

> **Superseded for live use.** IMAGE mode is the wrong mode for a loop — see
> [phase-6](phase-6-live.md), where VIDEO mode measures **12.17 ms on moving
> frames** and finds the hand **96 %** of the time against IMAGE mode's **37 %**.

## 6. Coordinate systems, verified not trusted

| documented claim | measured | verdict |
|---|---|---|
| normalized `z` origin is **the wrist** | `-3.3e-07`, `+2.8e-07` | ✅ |
| world landmarks are **metric** | palm span 9.91 / 9.44 cm (adult ≈ 9–11) | ✅ |
| world origin is the "approximate **geometric center**" | see below | ⚠️ |

### What the world origin actually is

Distance of each candidate from the reported origin:

| candidate | hand 0 | hand 1 |
|---|---|---|
| **mean of the 4 MCP knuckles** | **0.44 cm** | **0.75 cm** |
| middle MCP | 0.98 | 1.02 |
| bounding-box centre | 1.62 | 1.82 |
| mean of all 21 landmarks | 2.65 | 2.54 |
| wrist | 9.34 | 8.66 |

**It is the palm centre, not the landmark centroid.** The obvious guess is
2.5 cm wrong *and* unstable — curl a finger and a 21-point centroid moves. The
knuckles are in the rigid slab, so Phase 2 builds its frame on them.

### The two outputs are not the same data rescaled

World/image pair-distance ratio, across all landmark pairs:

| hand | min | median | max | spread |
|---|---|---|---|---|
| 0 | 0.346 | 0.533 | 0.978 | **2.83×** |
| 1 | 0.289 | 0.686 | 1.077 | **3.72×** |

A pure rescale gives exactly **1.00×**. So `world` is a genuinely separate 3-D
reconstruction — **which is why orientation taken from it is trustworthy.**

**But neither gives absolute position.** Both origins sit on the hand. The
distance to the camera is simply absent.

## 7. Model noise between two hands

Same person, so anatomy should match:

| | hand 0 | hand 1 | difference |
|---|---|---|---|
| palm span | 9.91 cm | 9.44 cm | **5 %** |
| hand length | 17.07 cm | 15.49 cm | **10 %** |

They are not really 10 % different. That spread is **estimation error**, and
another reason `pinch_ratio` divides by palm span rather than thresholding
metres.

## Gate: PASSED

| requirement | result |
|---|---|
| landmarks reproducible | ✅ bit-identical, PNG-pinned |
| per-frame latency table | ✅ p90 11.76 ms |
| fixtures frozen | ✅ `landmarks_golden.npz`, 4.3 KB |
| suite runs with no camera and no mediapipe | ✅ |
