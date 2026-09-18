# ======================================================================
# VENDORED from the phi repo -- do NOT hand-edit.
#   source : phi/simulation/so101_inverse_kinematics.py
#   commit : 66b919e
#   copied : 2026-09-16
#
# Verified in phi before copying:
#   FK agrees with mj_forward to 8.98e-09 m over 500 random poses
#   IK is exact to 0.00 pm over 3000 poses; enumerates all branches
#
# tests/test_kinematics_golden.py pins this copy to the phi original.
# If phi's version changes, that test fails. That is the point.
# ======================================================================
"""Closed-form inverse kinematics for the SO-101.

Given where you want the gripper, what are the joint angles?

This arm admits an EXACT algebraic solution -- no Jacobian, no iteration, no
local minima -- because of one structural fact:

    shoulder_lift, elbow_flex and wrist_flex are exactly parallel
    (verified from the model: |axis . axis| = 1.000000000 for all three pairs)

Three parallel joints = a PLANAR chain. So the arm is

    [ shoulder_pan ]  ->  [ planar 3R in the pan plane ]  ->  [ wrist_roll ]

and the problem splits into two small ones that each have a schoolbook answer:

    1. shoulder_pan is FORCED by the target position (the pan plane must
       contain the target). Two solutions: reach forwards or backwards.
    2. inside that plane, a 3R arm hitting (x, z) + a tool pitch is
       3 equations in 3 unknowns. Law of cosines. Two solutions: elbow
       up or down.

So up to 4 solutions per target. All are returned; the caller picks.

WHAT YOU CANNOT ASK FOR: tool YAW. With the approach axis locked into the pan
plane, yaw is whatever the position gives you. The arm is 5-DOF and a pose is
6-DOF; this is the DOF you lose, and you do not get to choose which.

Position is independent of wrist_roll (verified: 0.000e+00 m over 300 random
rolls) because the tool offset lies ALONG the roll axis. So roll is free.
"""

from __future__ import annotations

import numpy as np

from .forward import (get_forward_kinematics, get_g12, get_g23,
                                      get_g34, get_g45, get_g5t)

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]

LIMITS_DEG = {
    "shoulder_pan": (-110.0, 110.0), "shoulder_lift": (-100.0, 100.0),
    "elbow_flex": (-96.8, 96.8), "wrist_flex": (-95.0, 95.0),
    "wrist_roll": (-157.2, 162.8),
}

_D = np.array([0.0388353, 0.0, 0.0624])   # a point on the shoulder_pan axis
_RY180 = np.diag([-1.0, 1.0, -1.0])       # the fixed flip inside get_gw1


def _planar_geometry():
    """Measure the planar 3R off the forward kinematics, once."""
    A = get_g12(0.0); B = A @ get_g23(0.0); C = B @ get_g34(0.0)
    P2, P3, P4 = A[:3, 3], B[:3, 3], C[:3, 3]
    PT = (C @ get_g45(0.0) @ get_g5t())[:3, 3]
    pr = lambda v: np.array([v[0], v[2]])          # project out the pitch axis
    seg = lambda a, b: (np.linalg.norm(pr(b) - pr(a)),
                        np.arctan2(-(pr(b) - pr(a))[1], (pr(b) - pr(a))[0]))
    (L1, a1), (L2, a2), (L3, a3) = seg(P2, P3), seg(P3, P4), seg(P4, PT)
    return pr(P2), L1, L2, L3, a1, a2, a3, PT[1]


_O, _L1, _L2, _L3, _A1, _A2, _A3, _LAT = _planar_geometry()


def tool_pitch(shoulder_lift, elbow_flex, wrist_flex) -> float:
    """The tool's pitch angle in the pan plane, radians. The FK of the sum."""
    q = np.deg2rad([shoulder_lift, elbow_flex, wrist_flex])
    return _A3 + q.sum()


def _wrap(rad): return (np.rad2deg(rad) + 180.0) % 360.0 - 180.0


def inverse_kinematics(target_xyz, pitch_rad, roll_deg=0.0, verify=True):
    """All exact solutions putting the object frame at `target_xyz`.

    pitch_rad -- tool pitch in the pan plane (see tool_pitch()).
    Returns a list of {joint: degrees} dicts, possibly empty.
    """
    u = _RY180 @ (np.asarray(target_xyz, float) - _D)
    A, B, C = u[1], -u[0], _LAT
    h = np.hypot(A, B)
    if h < abs(C):                      # target too close to the pan axis
        return []

    out = []
    for pan_branch in (+1, -1):         # reach forwards / backwards
        th = np.arctan2(B, A) + pan_branch * np.arccos(np.clip(C / h, -1.0, 1.0))
        x1 = np.cos(th) * u[0] + np.sin(th) * u[1]
        z1 = u[2]

        # step back along the tool to the wrist_flex pivot
        wx = x1 - _L3 * np.cos(pitch_rad)
        wz = z1 + _L3 * np.sin(pitch_rad)
        ux, uz = wx - _O[0], wz - _O[1]
        reach = np.hypot(ux, uz)
        if reach > _L1 + _L2 or reach < abs(_L1 - _L2):
            continue                    # out of the 2R annulus

        for elbow in (+1, -1):          # elbow up / down
            cos_b = (reach**2 - _L1**2 - _L2**2) / (2 * _L1 * _L2)
            b = np.arccos(np.clip(cos_b, -1.0, 1.0)) * elbow
            base = (np.arctan2(-uz, ux)
                    - np.arctan2(_L2 * np.sin(b), _L1 + _L2 * np.cos(b)))
            t2 = base - _A1
            t3 = b - (_A2 - _A1)
            t4 = (pitch_rad - _A3) - t2 - t3
            sol = dict(zip(JOINTS, [_wrap(th), _wrap(t2), _wrap(t3), _wrap(t4),
                                    float(roll_deg)]))
            if verify:
                err = np.linalg.norm(get_forward_kinematics(sol)[0]
                                     - np.asarray(target_xyz, float))
                if err > 1e-9:
                    continue
            out.append(sol)
    return out


def within_limits(sol) -> bool:
    return all(LIMITS_DEG[k][0] <= v <= LIMITS_DEG[k][1] for k, v in sol.items())


def pick(solutions, current=None):
    """Choose one solution: in-limits, and closest to where we already are.

    Choosing by proximity is not cosmetic. Different branches are far apart in
    joint space, so switching branches between frames makes the real arm slam
    from one posture to another.
    """
    ok = [s for s in solutions if within_limits(s)]
    if not ok:
        return None
    if current is None:
        return ok[0]
    cost = lambda s: max(abs(s[j] - current[j]) for j in JOINTS)
    return min(ok, key=cost)


if __name__ == "__main__":
    rng = np.random.default_rng(7)
    errs, misses, branches = [], 0, []
    for _ in range(3000):
        q = rng.uniform(-80, 80, 3); pan = rng.uniform(-100, 100)
        ref = dict(zip(JOINTS, [pan, *q, 0.0]))
        p, _ = get_forward_kinematics(ref)
        sols = inverse_kinematics(p, tool_pitch(*q))
        if not sols:
            misses += 1
            continue
        branches.append(len(sols))
        errs.append(min(np.linalg.norm(get_forward_kinematics(s)[0] - p) for s in sols))
    errs = np.array(errs)
    print(f"round trip over 3000 poses: solved {len(errs)}, no solution {misses}")
    print(f"  position error  max {errs.max()*1e12:.2f} pm")
    print(f"  branches found  {np.bincount(branches)[1:]}  (1,2,3,4)")
    print("PASS" if errs.max() < 1e-9 and misses == 0 else "FAIL")
