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
