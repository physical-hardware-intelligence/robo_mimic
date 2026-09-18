# mirror docs

Three kinds of document, kept apart on purpose.

## Decisions — *why it is built this way*

Each one states the facts that forced a choice, what the choice costs, and the
test that would fail if it were violated.

| | |
|---|---|
| [ADR-001](adr/001-five-dof-projection.md) | Project 6-DOF hand poses onto 5 by discarding tool yaw — and **report** it |
| [ADR-002](adr/002-incremental-position-clutch.md) | Orientation absolute, position incremental. One camera cannot measure distance |
| [ADR-003](adr/003-clutch-on-a-key-gripper-on-the-pinch.md) | Clutch on a key, gripper on the pinch, orientation off, depth de-rated |

[Template](adr/000-template.md) for new ones.

## Measurements — *the numbers, and where they came from*

**This is the canonical home for every figure.** Code comments and the README
cite these; they do not restate them.

| | |
|---|---|
| [phase-1](measurements/phase-1-perception.md) | Perception. Version pinning, the JPEG-decoder trap, the noise floor |
| [phase-2](measurements/phase-2-handframe.md) | Hand frame + clutch. Orthonormality, the handedness mirror |
| [phase-3](measurements/phase-3-projection.md) | Pose → joints. The 5-DOF constraint as one equation |
| [phase-4](measurements/phase-4-safety.md) | The safety envelope. 30 000 adversarial frames, 0 escapes |
| [phase-5](measurements/phase-5-sim.md) | Sim in the loop. Interpolation beats the staircase 13.3× |
| [phase-6](measurements/phase-6-live.md) | Live. Per-stage budget, and two benchmarks of mine that were wrong |

## Code — *the intuition needed to read it*

Module docstrings carry the reasoning; they link here for the figures.
Start with [`src/mirror/__init__.py`](../src/mirror/__init__.py) for the
pipeline, then follow it left to right.

---

## Reading the measurements honestly

Each file separates three things, because they are not interchangeable:

- **measured** — a number from a run, with the conditions stated
- **[UNVERIFIED]** — believed but untested, usually because it needs hardware
- **corrections** — claims of ours that turned out to be wrong, kept rather than
  quietly deleted, because the wrong number often explains a design choice

Six corrections are recorded so far. They are the most useful pages here.
