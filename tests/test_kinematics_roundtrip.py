"""The round-trip invariant: IK undoes FK.

Measured before these bounds were chosen (500 trials each):
    joint-space recovery, best branch : 3.21e-11 deg
    task-space position, ALL branches : 5.19e-16 m
The tolerances below sit ~100x looser, so they cannot flake, but any real
regression moves the error by orders of magnitude and trips them.
"""

from __future__ import annotations

import numpy as np
import pytest

from robo_mimic.kinematics.forward import get_forward_kinematics
from robo_mimic.kinematics.inverse import inverse_kinematics, pick, tool_pitch, within_limits
from robo_mimic.kinematics.limits import JOINTS, LIMITS_DEG

JOINT_TOL_DEG = 1e-9
POSITION_TOL_M = 1e-13
TRIALS = 300


def _random_joints(generator: np.random.Generator) -> dict[str, float]:
    return {j: float(generator.uniform(*LIMITS_DEG[j])) for j in JOINTS}


def _fk(q: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    return get_forward_kinematics({**q, "gripper": 0.0})


def _travel(solution: dict[str, float], current: dict[str, float]) -> float:
    """Largest single-joint move from `current` to `solution`, in degrees."""
    return max(abs(solution[j] - current[j]) for j in JOINTS)


def test_some_branch_recovers_the_original_joints(rng: np.random.Generator) -> None:
    worst = 0.0
    for _ in range(TRIALS):
        q = _random_joints(rng)
        position, _ = _fk(q)
        pitch = tool_pitch(q["shoulder_lift"], q["elbow_flex"], q["wrist_flex"])
        solutions = inverse_kinematics(position, pitch, q["wrist_roll"])
        best = min(max(abs(s[j] - q[j]) for j in JOINTS) for s in solutions)
        worst = max(worst, best)
    assert worst < JOINT_TOL_DEG, f"worst joint recovery {worst:.3e} deg"


def test_every_branch_lands_on_the_same_point(rng: np.random.Generator) -> None:
    """Branches differ in posture, never in where the tool ends up."""
    worst = 0.0
    for _ in range(TRIALS):
        q = _random_joints(rng)
        position, _ = _fk(q)
        pitch = tool_pitch(q["shoulder_lift"], q["elbow_flex"], q["wrist_flex"])
        for solution in inverse_kinematics(position, pitch, q["wrist_roll"]):
            achieved, _ = _fk(solution)
            worst = max(worst, float(np.linalg.norm(achieved - position)))
    assert worst < POSITION_TOL_M, f"worst task-space error {worst:.3e} m"


def test_branch_count_is_two_or_four(rng: np.random.Generator) -> None:
    """2 pan branches x 2 elbow branches; degenerate cases collapse to 2."""
    counts = set()
    for _ in range(TRIALS):
        q = _random_joints(rng)
        position, _ = _fk(q)
        pitch = tool_pitch(q["shoulder_lift"], q["elbow_flex"], q["wrist_flex"])
        counts.add(len(inverse_kinematics(position, pitch, q["wrist_roll"])))
    assert counts <= {2, 4}, f"unexpected branch counts {counts}"


def test_pick_returns_an_in_limits_solution_or_none(rng: np.random.Generator) -> None:
    for _ in range(TRIALS):
        q = _random_joints(rng)
        position, _ = _fk(q)
        pitch = tool_pitch(q["shoulder_lift"], q["elbow_flex"], q["wrist_flex"])
        chosen = pick(inverse_kinematics(position, pitch, q["wrist_roll"]), current=q)
        if chosen is not None:
            assert within_limits(chosen)


def test_pick_prefers_the_nearer_branch(rng: np.random.Generator) -> None:
    """Branch continuity is a safety property: switching branches slams the arm."""
    for _ in range(TRIALS):
        q = _random_joints(rng)
        position, _ = _fk(q)
        pitch = tool_pitch(q["shoulder_lift"], q["elbow_flex"], q["wrist_flex"])
        solutions = inverse_kinematics(position, pitch, q["wrist_roll"])
        chosen = pick(solutions, current=q)
        if chosen is None:
            continue
        legal = [s for s in solutions if within_limits(s)]
        assert _travel(chosen, q) == pytest.approx(
            min(_travel(s, q) for s in legal), abs=1e-9
        )


def test_unreachable_target_does_not_crash() -> None:
    """A point two metres away is outside a 0.3 m arm. Expect no solution, not a traceback."""
    solutions = inverse_kinematics(np.array([2.0, 0.0, 0.0]), 0.0, 0.0, verify=False)
    assert pick(solutions) is None or all(
        not within_limits(s) for s in solutions
    ), "an unreachable point produced an in-limits solution"
