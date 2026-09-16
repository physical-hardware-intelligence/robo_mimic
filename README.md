# mirror

**Your hand moves. The arm mirrors it.**

Webcam hand-pose teleoperation for the [SO-101](https://github.com/TheRobotStudio/SO-ARM100) arm.
One RGB camera, no depth sensor, no gloves, no markers.

Built by [Φ — Physical Hardware Intelligence](https://github.com/physical-hardware-intelligence/phi),
a student robotics SIG at Northeastern University's Silicon Valley campus.

> **Status: Phase 0.** Scaffold, kinematics, and the gate. No perception yet. See [Phases](#phases).

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

## Quick start

```bash
make setup     # uv venv + all extras
make check     # THE GATE: lint + strict types + the pure-math suite
```

`make check` needs no camera, no robot, and no MuJoCo.

## Phases

Each one exits on a **committed number**, not on "it works."

| # | phase | exit gate |
|---|---|---|
| **0** ✅ | scaffold, kinematics, limits | `make check` green, real tests, ADR-001/002 written |
| 1 | perception, offline | landmarks from committed clips; per-frame latency table; fixtures frozen |
| 2 | hand frame + retarget | orthonormality under fuzz; clutch state machine table-covered |
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

**MediaPipe is not bit-for-bit reproducible across versions and platforms.** So the fixtures
store **its output, not its input**: everything downstream is deterministic even though
MediaPipe is not. MediaPipe itself gets a separate tolerance-based test.

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
