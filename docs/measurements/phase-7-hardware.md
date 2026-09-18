# Phase 7 — the real arm

Measured 2026-09-18 on our SO-101 follower, lerobot 0.6.0, calibration
`phi_follower.json`. Bring-up only: the hand-to-arm loop is built and its link
is proven, but has not yet driven the hardware end to end. Marked below.

## The rig

| | |
|---|---|
| follower | `/dev/tty.usbmodem5B7B0096441` — the one we drive |
| leader | `/dev/tty.usbmodem5B7B0139311` — **never** drive this; it is the input device |
| supply | **11.8–12.0 V** measured (see the contradiction below) |
| idle temperature | 31–33 °C |
| servos | STS3215 ×6, ids 1–6, 4096 ticks/turn |

**Two processes, on purpose.** Perception runs in our venv, the bus in phi's
conda env, joined by UDP ([link.py](../../src/robo_mimic/link.py)). Neither
environment can host both halves without damage, and both directions were
measured with `uv pip install --dry-run` rather than guessed:

| direction | cost |
|---|---|
| lerobot → our venv | **+36 packages including torch 2.11 + torchvision** |
| mediapipe → phi's env | a **second OpenCV** (5.0 over the existing 4.13) in the env the club's data collection runs on |

The split is also just right: perception manages ~20 Hz, the bus wants 50+.
Coupling them lets the slower set the rate for both. Same two-loop split that
bought [13.3× in sim](phase-5-sim.md), with a socket in the seam.

## Four things about lerobot that change the design

**1. In DEGREES mode it does not clamp on write.** The `RANGE_0_100` and
`RANGE_M100_100` paths bound the value. The `DEGREES` path is a bare
`int(val * 4095 / 360 + mid)` straight into `Goal_Position`. **For the five
body joints our `safety.py` clamp is the only guard.** In sim it was a nicety.

**2. Its degree zero is the midpoint of the calibrated range**, not the URDF's
kinematic zero. Nothing makes those agree. On our arm they do, checked by FK:

| pose | our FK says the tool is at |
|---|---|
| measured rest | reach **+14.45 cm**, lateral +0.88, height **−1.35 cm** |
| all joints zero | reach +39.66 cm, height +23.44 cm |

14.5 cm out and level with the base is exactly where a folded SO-101 sits, so
the conventions line up. That is a property of *this* calibration, so `check`
re-verifies it rather than trusting it.

**3. Our limits are a strict subset of this arm's calibrated travel** — the
safe direction, on every joint:

| joint | we command | arm has | margin |
|---|---|---|---|
| shoulder_pan | 220.00° | 230.68° | +10.68 |
| shoulder_lift | 200.00° | 208.79° | +8.79 |
| elbow_flex | 193.66° | 194.02° | **+0.36** |
| wrist_flex | 190.00° | 205.89° | +15.89 |
| wrist_roll | 320.00° | 360.00° | +40.00 |

Elbow has 0.36° of margin. That is thin enough to be worth knowing.

**4. `connect()` energises.** `configure()` runs inside `torque_disabled()`,
whose `finally` calls `enable_torque()`. Connecting is a physical act.

## The arm does not power up inside our envelope

It rests with `shoulder_lift` at **−104.04°** against our −100° limit, so the
first clamped command is **4.04° of motion nobody asked for**. `entry_pose()`
returns that number so the caller ramps into it at 15 °/s instead of jumping.

## Stale goal: a 106° trap, found by the gate

Enabling torque drives every servo to whatever `Goal_Position` it still holds
from the last session. Mid-bring-up an imitation-learning rollout ran on the
same arm, ended, and left the arm limp and sagging away from the policy's last
commanded pose:

| joint | present | goal | gap |
|---|---|---|---|
| shoulder_lift | 813 | 2022 | **+106.29°** |
| elbow_flex | 3016 | 2112 | **−79.47°** |
| wrist_roll | 1306 | 2046 | **+65.05°** |

Energising would have thrown three joints at once. **`check` refused.** This
was not a staged test; a real second process left a real trap.

`arm.py align` is the cure: `Goal_Position := Present_Position` with torque
still off, so nothing can move. Measured **106.29° → 0.00°**. `park` now does
it on the way out, so we stop leaving the trap for the next person.

> A limp arm has no fixed resting position — it sags, and any touch moves it.
> So stale goal **recurs by design** and cannot be engineered away. The gate
> plus `align` is the answer, not a tighter park.

## Holding error tracks gravity, exactly

Each joint jogged 10° at 15 °/s, then the reached position read back:

| joint | error | gravity load on that axis |
|---|---|---|
| wrist_roll | **−0.24°** | none — rotates about its own axis |
| shoulder_pan | **−0.42°** | none — rotates about vertical |
| wrist_flex | **+1.03°** | the gripper |
| elbow_flex | **+2.79°** | forearm + gripper |

Monotonic in load, which is the signature of **proportional-only control**.
lerobot sets `P_Coefficient=16` (half the servo default of 32), `I=0`, `D=32`,
with the comment *"to avoid shakiness"*. With no integral term, steady-state
error settles where `kp · error = gravity torque`, so error is proportional to
load. Exactly the `kp`/`kv` story from [mujoco-so101-setup], now on metal.

**The trade is explicit**: doubling `kp` would halve the sag and bring back the
17.9 Hz ring we characterised. Sag was chosen over shake. Worth re-deciding
with a number rather than inheriting it.

### What it costs at the tool

Those four errors applied at the measured pose, through our FK:

| | |
|---|---|
| tool displacement | **16.43 mm** |
| sim tracking (phase 5) | 0.442 mm p95 |
| **sim-to-real gap** | **37×** |

This is the headline number of the phase. The sim has a perfect position
servo; the real one sags under its own weight. No amount of better perception
closes it — it is a control problem, and an integral term or gravity
compensation is where the 16 mm goes.

## Two bugs the first live session found

Reported: *"gripper opens and closes, but the arm does not move left or right."*
The gripper working while the arm did not is the whole clue -- openness is
absolute and pose-independent, the body joints are not.

### 1. Perception was anchored to the sim, not to the arm

`teleop.py` did `limiter.reset(START)` against a hardcoded sim home. Position
is incremental (ADR-002), so **the starting pose IS the frame**:

| joint | sim START | real rest | gap |
|---|---|---|---|
| shoulder_lift | −40.00 | −104.04 | **+64.04** |
| wrist_flex | −25.00 | +77.01 | **−102.01** |
| wrist_roll | 0.00 | −71.78 | +71.78 |

**Tool frames 28.7 cm apart.** The arm spent the session crawling toward a pose
nobody asked for while a few degrees of hand-driven pan were lost in the noise.

Fixed with a reverse channel: `serve` reports `Present_Position` on udp 47102,
and teleop seeds the limiter and the sim from it. If no report arrives it
**refuses to start** -- silently falling back to `START` is the bug itself.

### 2. The rate limiter ramped from measured position, not from a setpoint

```python
goal = present + clamp(target - present, ±step)      # wrong
```

A servo under gravity needs a **standing** position error to hold at all: 2.79°
on the elbow. Commanding `present + 0.5°` hands it 18% of the error it needs
merely to stay put, so a loaded joint sags while you believe you are raising
it, and the setpoint can never get ahead of actual. Now an internal setpoint is
ramped and written, re-seeded from actual whenever the arm freezes.

This is the phase-5 `advance()` bug (see [phase-5](phase-5-sim.md)) reproduced
on hardware, where it costs more. **Fixing a bug in sim does not fix it in the
next implementation of the same idea.**

### Verified after the fix

| commanded | reached | error |
|---|---|---|
| shoulder_pan −40° | −39.6° | +0.4° |
| shoulder_pan +40° | +39.9° | −0.1° |
| shoulder_pan 0° | +0.5° | +0.5° |

80° of travel tracked to under half a degree. The elbow settled 3.0° below its
commanded 60°, independently reproducing the 2.79° sag measured by jogging.

## Where to start a session, and why not where it rests

The arm's own gravity rest -- folded, jaw down -- is the obvious start and is
disqualified three times over:

| pose | reach | height | manipulability | limit margin |
|---|---|---|---|---|
| gravity rest, jaw down | 14.4 cm | **−1.4 cm** | **1.51e-08** | **−4.0°** |
| **READY** | 26.3 cm | +12.0 cm | 3.07e-08 | +36.8° |
| old sim START | 31.2 cm | +22.0 cm | 4.82e-08 | +41.8° |

It sits **4° outside** our shoulder_lift limit, so we cannot command it. Its
tool is **below the shoulder**, so there is nowhere to reach down to, and
reaching down is most of teleoperation. And it has the **lowest manipulability
measured**, because a folded arm sits near its workspace edge where hand motion
buys little arm motion.

`READY` is mid-range everywhere: twice the manipulability and 12 cm of descent.
It is deliberately **not** the most manipulable pose available -- the old sim
start scored higher but holds the arm high and extended, a longer fall if
torque drops.

The arm can neither reach READY from limp nor hold it unpowered, so
`serve --start ready` drives it there and returns it to the folded rest on exit.

> **Park must end at a pose the arm can hold with no torque.** Returning to
> "where I found it" is only safe if it was found limp. Found energised, it may
> be holding a pose it cannot hold unpowered, and cutting torque drops it.
> `Arm` now checks `Torque_Enable` at startup and folds down instead.

Also: SIGTERM now parks. A `pkill` left the arm energised for ten minutes.

## Smoothness: the filter was in the wrong space

Reported: *"it is mirroring instead of mimicking"*, and *"pick and place jitters
and makes the robot unpredictable"*. Two unrelated causes.

### Mirror vs mimic is a convention, not a bug

The preview is flipped so your hand reads as a reflection -- the usual webcam
courtesy, and right for watching your own hand. That flip was then inherited by
the arm, which is a separate question and was never decided deliberately.
Facing the arm, your left should move the tool to YOUR left. Default is now
**mimic**; `teleop --mirror` restores the reflection, which is correct if you
stand BEHIND the arm facing the way it faces.

Unlike vertical (image y grows downward) and depth (the 1/span proxy shrinks
with proximity), the lateral sign is **not** determined by physics, so the
tests now read it from the config and pin the convention separately.

### The filter was smoothing joints, after IK had already jumped

`safety.py` filters joint angles. But IK is sharply nonlinear: a centimetre of
depth noise near a workspace edge lands on a different pitch, tens of degrees
away, and the joint filter then faithfully smooths its way toward a pose nobody
wanted. **Smoothing after the nonlinearity cannot undo it.**

Simulated pick -- descend 12 cm, dwell, lift -- at 20 Hz with this project's
measured perception noise (1.5 mm lateral, 15 mm depth, depth being 7-25x worse
because the proxy divides by span SQUARED):

| configuration | jumps >15° | worst step | pitch shifts | lag |
|---|---|---|---|---|
| joint filter only | 30.4 | **62.30°** | 2.0 | 12.0 mm |
| + Cartesian fc=2.0 | 0.8 | 15.32° | 0 | 7.1 mm |
| + Cartesian fc=1.0 | **0** | 9.22° | 0 | 8.8 mm |
| + Cartesian fc=1.0, depth ×0.4 | **0** | **5.34°** | 0 | **7.6 mm** |
| + Cartesian fc=0.5, depth ×0.4 | 0 | 4.41° | 0 | 13.2 mm |

**11.7× smoother and more accurate at the same time.** Lag falling looks wrong
for a filter until you notice the noise was itself most of the tracking error.
fc=0.5 is smoother still and pays for it in lag; 1.0 is the knee.

### What the whole pipeline was actually doing

With the joint filter and rate limiter downstream, over the same pick:

| | before | after |
|---|---|---|
| frames pinned at the 120 °/s cap | **51.9%** | 3.4% |
| frames dropped by the glitch guard | 40.4 | **0** |
| tracking lag | 31.8 mm | **10.1 mm** |
| tool jitter | 13.71 mm/frame | **4.43 mm/frame** |

The arm spent **more than half of every session at its velocity limit**,
chasing a target that kept teleporting, while the glitch guard discarded 8% of
frames outright. It was never tracking the hand; it was permanently catching
up. That is the whole of "jittery and unpredictable".

> The velocity cap and the glitch guard are safety features and both were
> working. Saturating them continuously is not a safety event, so nothing
> complained -- the readout said "limited", which is what it is for. Worth
> surfacing saturation as a warning, not just a state.

Depth is de-rated to 0.4× (0.035 → 0.014 m/span) because it is both the
noisiest axis and the direction the arm is worst conditioned in. The cost is
honest: reaching forward now takes more hand travel.

## Contradiction to resolve

The wiki's sim-to-real note says *"1.75× stronger than our 7.4 V arm
(3.35 vs 1.91 N·m)"*. **Measured supply is 11.8–12.0 V.** Which STS3215 variant
is fitted is **[UNVERIFIED]**, but the 7.4 V premise that torque comparison
rests on is contradicted by the bus, and should not be requoted until checked.

## What is verified, and what is not

| | |
|---|---|
| ✅ read-only pre-flight, 12 checks | green on hardware |
| ✅ energise without snapping | goal := present first, then torque |
| ✅ hold under gravity | 12 s, drift settled at 1.41° |
| ✅ single-joint sign and scale | +10.00° commanded, −0.24° error |
| ✅ park and align on exit | 106.29° → 0.00° |
| ✅ the UDP link | 300 sent, 300 received, 0 out of order |
| ⬜ **`serve` driving the arm** | written, not yet run: the arm was in use |
| ⬜ **hand → real arm, end to end** | the remaining gate for this phase |
| ⬜ velocity caps `120 °/s`, `2.5 /s` | still **[UNVERIFIED]** against the STS3215 |

## Running it

```bash
# phi's env owns the bus
~/miniforge3/envs/phi/bin/python scripts/arm.py check     # read-only
~/miniforge3/envs/phi/bin/python scripts/arm.py align     # disarm a stale goal
~/miniforge3/envs/phi/bin/python scripts/arm.py serve --speed 20

# our venv owns perception, in a second terminal
make teleop ARGS="--arm" CAMERA=1
```

`serve` freezes the arm if datagrams stop for 250 ms, and parks on ctrl-c.
The clutch still gates everything: `SPACE` off means the arm holds.

[mujoco-so101-setup]: https://github.com/physical-hardware-intelligence/phi
