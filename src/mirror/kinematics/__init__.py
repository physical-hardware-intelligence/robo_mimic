"""SO-101 kinematics.

`forward` and `inverse` are VENDORED from the phi repo -- see the header in each
file. They are excluded from ruff and mypy on purpose: linting vendored code
means editing it, and editing it means the provenance is a lie.

`limits` is ours, and exists because phi's hand-typed limit table is slightly
wider than the model actually allows.
"""

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
