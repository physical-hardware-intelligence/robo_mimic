# ADR-002: Map orientation absolutely, position incrementally through a clutch

**Status**: Accepted; the clutch mechanism is superseded by [ADR-003](003-clutch-on-a-key-gripper-on-the-pinch.md)
**Date**: 2026-09-16

> The core finding here stands: one camera cannot measure absolute distance, so
> position must be incremental. What changed is what ENGAGES that increment.
> ADR-003 moves the clutch from a pinch to a held key, freeing the pinch for the
> gripper, and de-rates the depth axis after measuring it at 7-25x the noise.

## Context

MediaPipe's hand landmarker reports 21 landmarks in two coordinate systems. The docs' exact
wording matters here, so it is quoted rather than paraphrased:

**`hand_landmarks`** — "`x` and `y` are normalized to `[0.0, 1.0]` by the image width and
height respectively." "`z` represents the landmark depth **with the depth at the wrist being
the origin**, and the smaller the value the closer the landmark is to the camera. The
magnitude of `z` uses roughly the same scale as `x`."

**`hand_world_landmarks`** — "`x`, `y` and `z`: Real-world 3D coordinates in meters **with the
origin at the hand's approximate geometric center**."

Source: MediaPipe Hands solution documentation.

**Both origins sit on the hand.** One is the wrist, the other the palm centroid. Neither is
the camera. So:

- **Orientation is fully recoverable.** `hand_world_landmarks` are metric, and the palm
  (`WRIST`, `INDEX_MCP`, `PINKY_MCP`) is nearly rigid, so an orthonormal frame built from
  those three points is stable and physically meaningful.
- **Absolute position is not recoverable.** The image gives a *bearing* from the camera. The
  distance along that bearing is absent.

Distance could in principle be inferred from apparent hand size, but hand sizes vary by
person, so this is a per-operator calibration problem wearing a feature's clothing.

## Decision

- **Orientation: absolute.** Hand orientation maps directly to tool orientation.
- **Position: incremental, through an explicit clutch.** Pinch to engage; hand motion becomes
  scaled tool motion; release to freeze. Like lifting a mouse off the desk.

## Consequences

**Buys**:
- The depth problem disappears rather than being approximated. Only *changes* in the image
  bearing are used, which are well conditioned even when absolute depth is not.
- Unlimited effective workspace. Clutch, reposition, re-engage, exactly as with a mouse.
- No per-operator calibration. Hand size cancels out of a differential mapping.

**Costs**:
- The operator must learn the clutch. It is one gesture, and it is the same idea as a mouse.
- Drift accumulates over a long session, so "return to home" must be a first-class command.
- `retarget.py` becomes **stateful** (it holds the engagement anchor), unlike every other
  pure stage. Its state machine is therefore table-tested exhaustively rather than
  spot-checked.

## Alternatives rejected

| alternative | why it lost |
|---|---|
| Absolute position from a calibrated box | Needs the depth that the sensor does not provide. Would require per-operator hand-size calibration and would still drift with distance. |
| Absolute position, depth from apparent hand size | Hand size varies by person by well over 10 percent. Turns a geometry problem into a biometrics problem. |
| Add a depth camera (RealSense D405/D435) | Solves it honestly, and phi's docs confirm RealSense is supported. Rejected **for now** as scope: the point is to work with one ordinary webcam. Revisit if the clutch proves unusable. |
| Orientation-only, position fixed | Safe and trivial, but not the project. Kept as the **Phase 5 first milestone** because it isolates orientation error from position error while validating. |

## How this is verified

- **Clutch state machine**: exhaustive table test over every transition, including engage
  while already engaged, release while released, and tracking loss in each state.
- **Scale invariance**: a synthetic hand scaled by 0.8x and 1.25x must produce the *same*
  tool trajectory, because a differential mapping is invariant to hand size. This is the
  test that proves the decision delivered what it promised.
- **Drift**: a closed-loop recorded trajectory (hand returns to its start) must leave the
  tool within a measured bound of its start. The bound is reported, not asserted to be zero.
