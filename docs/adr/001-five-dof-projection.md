# ADR-001: Project 6-DOF hand poses onto 5 DOF by discarding tool yaw, and report it

**Status**: Accepted, with a **factual correction** (2026-09-17) — see *Which axis, exactly*
**Date**: 2026-09-16

## Context

The SO-101 has five joints that move the tool (`shoulder_pan`, `shoulder_lift`,
`elbow_flex`, `wrist_flex`, `wrist_roll`) plus a jaw. Measured on the model:

| claim | test | result |
|---|---|---|
| five independent tool twists | rank of the 6x5 space Jacobian, 200 random in-limit poses | **rank 5 in 200/200** |
| the jaw is not an arm DOF | drive `gripper` 0 -> 1.5 rad, 2000 poses, watch the tool frame | **0.000e+00 m** |
| three pitch joints exactly parallel | pairwise \|axis . axis\| | **1.000000000** |
| `shoulder_pan` swings that plane | \|pan axis . plane normal\| | **1.45e-07** |
| `wrist_roll` lies in that plane | \|roll axis . plane normal\| | **1.53e-07** |
| the arm is *rigidly* planar | tool offset from the shoulder plane, 2000 poses | **18.1 mm, spread 0.000000 mm** |

The structure is `pan -> planar 3R -> roll`. Because the target position determines the pan
angle, it determines the arm plane, which fixes the tool's yaw. A hand pose supplies six
numbers; five of them are honourable requests.

The offset in the last row is not *small*, it is **constant**. The arm is not approximately
planar; it is a plane sitting 18.1 mm to one side.

## Decision

`project.py` maps a desired 6-DOF tool pose to the nearest achievable 5-DOF pose by
**discarding rotation about the arm-plane normal (tool yaw)** and keeping position, tool
pitch, and tool roll.

The discarded angle is **returned as a first-class value**, not swallowed. The operator UI
displays it live.

## Consequences

**Buys**: a total function. Every hand pose yields a valid arm command, with no unreachable
targets and no solver failures mid-motion.

**Costs**:
- Turning the hand left or right does nothing. This must be *taught*, which is why the
  discarded angle is on screen: the operator learns the arm's shape in about a minute
  instead of concluding the tracking is broken.
- Top-down approaches are a special case. When the approach axis is vertical the azimuth is
  undefined, so `wrist_roll` becomes a full 360 degrees of genuine tool yaw. **Top-down
  grasps have free yaw; nothing else does.**

## Alternatives rejected

| alternative | why it lost |
|---|---|
| Least-squares over all six DOF | Spreads the unreachable yaw into *position* error. Trades an honest 20 degrees of yaw error for centimetres of position error the operator cannot diagnose. |
| Reject unreachable poses | Every hand pose is unreachable in yaw. The arm would refuse to move. |
| Add a sixth joint | A hardware change, not a software decision. Out of scope. |
| Silently drop yaw | Cheapest to build, worst to use: the operator cannot tell a limitation from a bug. |

## How this is verified

**The residual must be a pure rotation about the arm-plane normal, and nothing else.**

```
project(T) differs from T by exactly Rot(plane_normal, yaw_discarded)
```

Tested over 10,000 random poses (Phase 3). If a bug leaks position error into the
projection, this invariant catches it; nothing else would.

Supporting invariant, already available: `FK(IK(project(T))) == project(T)` to IK precision.
Measured on the vendored modules: joint-space recovery **3.21e-11 deg**, task-space position
**5.19e-16 m**.


---

## Correction (2026-09-17): which axis, exactly

This ADR said the projection discards *"rotation about the arm-plane normal (tool yaw)"*. **The named axis was wrong.** Rotation about the plane normal **is the pitch**, which is fully achievable.

Measured over 1777 samples (154 IK branch flips rejected), at fixed position:

| | min | median | max |
|---|---|---|---|
| \|pitch axis · n̂\| | **1.0000000** | 1.0000000 | 1.0000000 |
| \|roll axis · n̂\| | **0.0000000** | 0.0000000 | 0.0000000 |
| \|**missing** · n̂\| | **0.0000000** | 0.0000000 | 0.0000000 |
| \|**missing** · â\| | **0.0000000** | 0.0000000 | 0.0000000 |

**The discarded axis lies IN the plane, perpendicular to the wrist axis**: `ĉ = n̂ × â`.

The *physical* description in this ADR was right all along — you cannot tilt the gripper out of the arm's plane, and the arm always points outward from its base. Only the algebra naming the axis was wrong.

### The constraint, restated correctly

Measured over 3000 poses: the tool frame's **x-axis is the wrist/roll axis** (unmoved by 37° of `wrist_roll`: `7.85e-17`) and it is **confined to the arm plane** (`0.000000000`, min = median = max). So the whole 5-DOF constraint is one scalar equation:

```
â · n̂ = 0
```

Everything in `project.py` follows from that line.
