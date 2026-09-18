# ADR-003: Clutch on a key, gripper on the pinch, orientation off, depth de-rated

**Status**: Accepted · **Date**: 2026-09-17 · Supersedes the clutch half of [ADR-002](002-incremental-position-clutch.md)

## Context

ADR-002 put the clutch on a thumb-index pinch. Four measurements since then say that was wrong, or at least incomplete.

**1. A pinch is the gripper's gesture, not the clutch's.** The SO-101's 6th motor is a jaw. Measured: driving it 0 → 1.5 rad moves the tool frame by **exactly 0.000e+00 m** — zero coupling to the arm. One gesture cannot mean both "engage motion" and "close the jaw", because you could never approach an object with the gripper open.

**2. Depth is an order of magnitude noisier than lateral.** `image_position = (cx/s, cy·a/s, 1/s)`. Differentiating: lateral error goes as `dcx/s`, depth as `ds/s²` — `s` **squared**. With `s ≈ 0.155`:

| camera noise σ | lateral | depth | ratio |
|---|---|---|---|
| 0.5 levels | 0.142 mm | 1.041 mm | 7.3× |
| 1.0 levels | 0.226 mm | **2.331 mm** | **10.3×** |
| 2.0 levels | 0.432 mm | **10.822 mm** | **25.0×** |

**3. The IK amplifies it, ~0.6°/mm.**

| tool jitter | worst joint jitter |
|---|---|
| 0.23 mm | 0.462° |
| 2.33 mm | **4.426°** |
| 10 mm | **19.631°** |

And the arm rings at **17.9 Hz** with **ζ = 0.250**, so broadband jitter lands on its resonance.

**4. Orientation is mostly unreachable.** For a position the arm can reach, sweeping pitch over its true range:

| position | reachable pitches |
|---|---|
| (0.219, −0.275, 0.294) | 15.5% |
| (0.208, −0.296, 0.359) | 8.0% |
| (0.058, 0.237, 0.375) | 37.0% |
| (0.282, −0.240, 0.368) | 7.0% |

**7–37%.** That is the missing 6th DOF, measured.

## Decision

| signal | drives |
|---|---|
| held key (`engage: bool`) | the clutch |
| hand position | tool position, incremental from engage |
| thumb-index gap | jaw opening, continuous, `[0,1]` |
| hand rotation | tool rotation — **off by default** |

Plus `depth_scale_m_per_span = 0.035` against `scale_m_per_span = 0.10`.

`Command.target` and `Command.gripper` are **both** `None` when the clutch is out. A deadman that still lets the jaw move is not a deadman.

## Consequences

**Buys**
- The pinch does its natural job, continuously (0.09 / 0.68 / 1.00, not a switch).
- A key cannot false-trigger the way a threshold can. **This is a safety device, not UX.**
- *Simpler*: no `pinch_on`/`pinch_off`, no hysteresis band, no chatter analysis. A boolean deletes that whole failure mode.
- Most commanded poses are now reachable, so the arm stops stuttering between solved and unsolvable.

**Costs**
- One hand on the keyboard.
- No wrist orientation until re-enabled. The students' project shipped without it.
- Depth feels less responsive than lateral. **That is the point** — it carries 10× the noise for the same information.

## Rejected

| alternative | why it lost |
|---|---|
| Keep pinch = clutch, gripper elsewhere | Leaves the jaw on an unnatural gesture and keeps the hysteresis machinery. |
| Pinch = clutch **and** gripper together | Cannot move with the gripper open. Rules out approach-then-grasp. |
| Fist = clutch | A fist *is* minimum openness. Same collision, worse ergonomics. |
| Second hand = clutch | Doubles tracking cost and occupies both hands. |
| No clutch at all (the students' design) | Needs absolute position, which one camera cannot give. Loses the deadman. |

## Verified by

- All 10 clutch transitions, table-tested, incl. `target` and `gripper` appearing/disappearing together.
- Jaw mapping clamped into the model's own `ctrlrange` under fuzz over ratios in [−10, 10].
- Jaw opening invariant to hand size (ratio divides out palm span), 200 hypothesis cases.
- `engage=False` never commands a pose or a jaw, over 200 random sequences.

## Open

**The viewer's SPACE is a toggle, not a held key.** OpenCV cannot detect key-hold: macOS sends one keydown, ~500 ms of nothing, then repeats. `Retargeter.step` takes `engage` as a plain boolean so the source can change without touching the logic. **Phase 7 needs a real momentary switch** — footswitch or gamepad trigger.
