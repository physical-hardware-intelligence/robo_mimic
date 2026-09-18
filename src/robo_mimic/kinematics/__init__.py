"""SO-101 kinematics.

`forward` and `inverse` are VENDORED from the phi repo -- see the header in each
file. They are excluded from ruff and mypy on purpose: linting vendored code
means editing it, and editing it means the provenance is a lie.

`limits` is ours, and exists because phi's hand-typed limit table is slightly
wider than the model actually allows.
"""

from typing import TYPE_CHECKING

from .limits import (
    CTRLRANGE_RAD,
    GRIPPER_RAD,
    JOINTS,
    LIMITS_DEG,
    clamp_deg,
    gripper_openness,
    gripper_rad,
    span_deg,
    within_limits_deg,
)

__all__ = [
    "CTRLRANGE_RAD",
    "GRIPPER_RAD",
    "JOINTS",
    "LIMITS_DEG",
    "clamp_deg",
    "gripper_openness",
    "gripper_rad",
    "span_deg",
    "within_limits_deg",
]


# --- typed boundary over the vendored, untyped modules -----------------------
# `forward.py` and `inverse.py` are phi's code and are excluded from mypy, so
# calling them from typed modules trips `no-untyped-call`. Wrapping them once
# here keeps every ignore in one place instead of scattered across callers.

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    F64 = npt.NDArray[np.float64]


def forward_kinematics(joints: dict[str, float]) -> "tuple[F64, F64]":
    """Joint angles (degrees, plus `gripper`) -> (tool position, tool rotation)."""
    from .forward import get_forward_kinematics

    position, rotation = get_forward_kinematics(joints)  # type: ignore[no-untyped-call]
    return position, rotation


def solve_ik(
    position: "F64", pitch_rad: float, roll_deg: float = 0.0, *, verify: bool = True
) -> list[dict[str, float]]:
    """All exact joint solutions putting the tool at `position`. May be empty."""
    from .inverse import inverse_kinematics

    out: list[dict[str, float]] = inverse_kinematics(  # type: ignore[no-untyped-call]
        position, pitch_rad, roll_deg, verify
    )
    return out


def in_limits(solution: dict[str, float]) -> bool:
    """True when every joint is inside its MODEL-DERIVED range.

    Deliberately NOT phi's `within_limits`, which consults the hand-typed table.
    The two disagree, and the disagreement bites: for `wrist_roll` the model
    allows -157.2110 deg while phi's table stops at -157.2, so a value clamped
    to the model's own limit is rejected by phi's check. `limits.py` exists to
    be the single source of truth; this is the function that enforces it.
    """
    return within_limits_deg(solution)


def tool_pitch_of(shoulder_lift: float, elbow_flex: float, wrist_flex: float) -> float:
    """The tool's pitch in the arm plane, radians, from the three pitch joints."""
    from .inverse import tool_pitch

    return float(tool_pitch(shoulder_lift, elbow_flex, wrist_flex))


__all__ += ["forward_kinematics", "in_limits", "solve_ik", "tool_pitch_of"]
