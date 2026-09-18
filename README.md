# mirror

**Your hand moves. The arm mirrors it.**

Webcam hand-pose teleoperation for the [SO-101](https://github.com/TheRobotStudio/SO-ARM100) arm.
One RGB camera, no depth sensor, no gloves, no markers.

Built by [Φ — Physical Hardware Intelligence](https://github.com/physical-hardware-intelligence/phi),
a student robotics SIG at Northeastern University's Silicon Valley campus.

> **Status: Phase 3 complete.** The pipeline now runs end to end to joint angles.
> `make check` = **162 tests, no camera required**. See [Phases](#phases).

---

## Two facts that shape everything here

Most hand-teleop projects discover these in week three. They are load-bearing, so they are
written down as decision records, not comments.

### 1. One RGB camera cannot tell you where a hand is

MediaPipe reports hand-**relative** geometry only:

| output | the docs' exact wording | what you actually get |
|---|---|---|
| `hand_landmarks` | "`x` and `y` are normalized to `[0.0, 1.0]` by the image width and height"; "`z` represents the landmark depth **with the depth at the wrist being the origin**" | a *direction* from the camera, plus hand-relative depth |
| `hand_world_landmarks` | "Real-world 3D coordinates in meters **with the origin at the hand's approximate geometric center**" | metric **shape and orientation**, no absolute location |

Both origins sit **on the hand**. Neither says how far away it is.

**So:** orientation maps absolutely. Position is **incremental, with a clutch** — pinch to
engage, move, release to freeze, like lifting a mouse off the desk. This sidesteps the depth
problem instead of pretending to solve it.
→ [ADR-002](docs/adr/002-incremental-position-clutch.md)

### 2. The arm has five degrees of freedom. A hand pose has six.

The whole constraint is **one scalar equation**, measured over 3000 poses:

```
â · n̂ = 0     â = the tool's x-axis (the wrist/roll axis),  n̂ = the arm-plane normal
```

The tool x-axis is unmoved by `wrist_roll` (**7.85e-17**) and never leaves the
plane (**0.000000000**, min = median = max). The rotation you cannot have is
about `ĉ = n̂ × â` — **in** the plane, perpendicular to the wrist.

The SO-101 is `pan → planar 3R → roll`. Its tool Jacobian has **rank 5 in 200/200** random
in-limit poses, so at every configuration there is a 1-D family of tool motions it simply
cannot produce.

**The discarded DOF is tool yaw**, and `mirror` reports it live rather than dropping it
silently — so the operator learns the machine's shape instead of fighting a mystery.

> Move your hand anywhere: the arm follows. Tilt it, twist it: the arm follows.
> **Turn it left and right: nothing happens.** The arm always points outward from its base.

→ [ADR-001](docs/adr/001-five-dof-projection.md)

---

## Architecture

Every stage between the two ends is a **pure function**. That is the whole reason the suite
runs in CI with no camera and no arm.

```
 camera ──► landmarks ──► hand frame ──► retarget ──► project ──► safety ──► IK ──► sink
[impure]   [mediapipe]     [pure]        [pure]       [pure]      [pure]   [pure]  [impure]
              │                                          │           │
         version-pinned;                        the 5-DOF drop   the fuzz target
         its OUTPUT is the fixture               lives HERE       (highest-ROI test)
```

## Try it

```bash
make setup     # uv venv (off exFAT) + all extras
make doctor    # deps, model, camera permission, pipeline -- one line each
make view      # live hand tracking from the built-in camera
```

**`SPACE` is the clutch.** Nothing moves until it is ON. With it on: move your
hand to move the tool, **pinch to close the jaw, spread to open it**.
`C` captures a clip, `R` resets, `Q` quits.

| on screen | what it is |
|---|---|
| green skeleton | the 21 landmarks |
| red / green / blue arrows | palm frame: x, y (fingers), z (out of palm) |
| CLUTCH | grey = off (nothing moves), green = on |
| JAW bar | gripper opening, 0 → 1 |
| PINCH bar | the raw ratio behind the jaw; ticks are closed/open |
| TOOL cm | where the arm would be commanded |

### What drives what ([ADR-003](docs/adr/003-clutch-on-a-key-gripper-on-the-pinch.md))

| signal | drives | why |
|---|---|---|
| held key | the clutch | a key cannot false-trigger; a gesture threshold can. **Deadman, not UX.** |
| hand position | tool position | incremental from wherever you engaged |
| thumb-index gap | **jaw opening**, continuous | the natural grasp gesture, and the jaw is measured to be **exactly decoupled** from the arm (0.000e+00 m) |
| hand rotation | tool rotation — **off** | only 7–37% of pitches are reachable on a 5-DOF arm |

**The jaw is gated by the clutch too.** A deadman that still lets the gripper
crush is not a deadman.

⚠️ `SPACE` is a **toggle**, not a held key: OpenCV cannot detect key-hold
(macOS sends one keydown, ~500 ms of silence, then repeats). The API takes
`engage` as a boolean so the source can change. **Hardware needs a real
momentary switch.**

**Camera permission is per-application.** macOS grants it to whichever app
launched the process, so run `make view` from the terminal you normally use and
click Allow. If no prompt appears: *System Settings → Privacy & Security →
Camera*.

```bash
make record NAME=wave           # SPACE to start/stop; saves .mp4 + landmark .npz
make replay CLIP=fixtures/clips/wave-<stamp>.mp4    # same pipeline, no camera
```

`make replay` re-runs a recorded clip through the identical code path, so a
result you saw live is reproducible without you in front of the lens.

## Verify

```bash
make check     # lint + strict mypy + 162 tests. No camera, no robot, no MuJoCo.
make test-all  # adds the 13 perception tests (needs mediapipe + make assets)
make cov       # coverage
```

## Phases

Each one exits on a **committed number**, not on "it works."

| # | phase | exit gate |
|---|---|---|
| **0** ✅ | scaffold, kinematics, limits | `make check` green, real tests, ADR-001/002 written |
| **1** ✅ | perception, offline | bit-identical landmarks, PNG-pinned; p90 **11.76 ms** (85 FPS). [measurements](docs/measurements/phase-1-perception.md) |
| **2** ✅ | hand frame + retarget + **gripper** | orthonormal to **1e-16**; mirror test exact; all 10 clutch transitions; jaw clamped under fuzz. [measurements](docs/measurements/phase-2-handframe.md) |
| 3 | the 5-DOF projection | residual proven to be a pure yaw rotation, 10k poses |
| 4 | safety layer | 10k fuzz cases, zero escapes |
| 5 | sim in the loop | tracking-error table + a side-by-side clip |
| 6 | live sim | per-stage latency budget, sustained FPS |
| 7 | hardware | `Present_Position` read first, return-to-start on exit, e-stop tested before motion |

## Tested, verified, validated

Three different things. Conflating them is how a project claims done while broken.

| | question | mechanism | where |
|---|---|---|---|
| **Tested** | does the code do what I wrote? | unit + integration over fixtures | CI, every push |
| **Verified** | does what I wrote satisfy the spec? | property + fuzz over invariants | CI, every push |
| **Validated** | is the spec the right spec? | measured latency, tracking error, a human operator | by hand, in a table |

**Fixtures store MediaPipe's output, not its input**, so every downstream stage is
deterministic in CI with no camera and no 60 MB native dependency.

Phase 1 sharpened *why*. Within a pinned version on one machine MediaPipe is **bit-identical**
(0.000e+00 over 30 calls and 5 fresh detectors), so goldens assert exact equality. But:

- **`mediapipe 1.0.1` aborts on macOS arm64** inside `TensorsToDetectionsCalculator`
  (the palm detector) — `delegate=CPU` does not help. Pinned to **`0.10.35`**.
- **The JPEG decoder changes the answer.** MediaPipe's loader and OpenCV's differ by ≤3
  levels on 2.79 % of pixels, and that moves world landmarks by **0.7370 mm**. The image
  *format* (SRGBA vs SRGB) changes nothing. Fix: transcode once to **PNG**, which is lossless
  and decoder-independent → **0.000000 mm**.
- **Noise floor**: σ = 1 level of image noise moves landmarks **2.6 mm**; σ = 4 moves them
  **11.2 mm**. Jitter of millimetres is intrinsic, which is why filtering is a safety
  component and why a *differential* position mapping beats an absolute one.

Full numbers: [docs/measurements/phase-1-perception.md](docs/measurements/phase-1-perception.md).

## Kinematics provenance

`src/mirror/kinematics/forward.py` and `inverse.py` are **vendored** from
`phi/simulation/` at commit `66b919e`. Verified there before copying:

- FK agrees with `mj_forward` to **8.98e-09 m** over 500 random poses
- IK is exact to **0.00 pm** over 3000 poses, and enumerates all branches

They are excluded from ruff and mypy on purpose: linting vendored code means editing it, and
editing it makes the provenance a lie. `tests/test_kinematics_golden.py` pins the copy to the
original — including a test that imports phi live and demands **bit-identical** output.

`limits.py` is ours. It exists because phi's hand-typed limit table is very slightly **wider
than the model permits** on three of five joints (`shoulder_pan`, `wrist_flex`, `wrist_roll`).
Harmless in a homework script; not harmless in the module that decides what a servo may do.
Our limits are derived from the model's `ctrlrange`, and a test re-parses the XML to prove it.

## License

Apache-2.0. Same as phi.
