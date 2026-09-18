r"""Joint limits for the SO-101, derived from the model, never retyped.

WHY THIS MODULE EXISTS
----------------------
phi's `so101_inverse_kinematics.py` carries a hand-typed `LIMITS_DEG` table.
Three of its five entries sit OUTSIDE the model's real `ctrlrange`:

    shoulder_pan   typed +-110.0    model +-109.9999   -> 1.1e-4 deg over
    wrist_flex     typed  +-95.0    model  +-94.9998   -> 2.3e-4 deg over
    wrist_roll     typed 162.8      model  162.7893    -> 1.1e-2 deg over

Harmless in a homework script. Not harmless in the module that decides what a
physical servo is allowed to do. So the numbers below are the EXACT radian
`ctrlrange` values from the model file, and degrees are computed, not typed.

PROVENANCE -- regenerate with:
    python -c "import re,pathlib; \
      print(re.findall(r'ctrlrange=\"([-\d. ]+)\"', \
      pathlib.Path('<phi>/simulation/model/so101_new_calib.xml').read_text()))"

tests/test_limits.py re-parses the XML when it is reachable and asserts these
match. If someone edits the model, the test says so.
"""

from __future__ import annotations

import math
from typing import Final

#: Exact `ctrlrange` radians, copied verbatim from so101_new_calib.xml.
CTRLRANGE_RAD: Final[dict[str, tuple[float, float]]] = {
    "shoulder_pan": (-1.91986, 1.91986),
    "shoulder_lift": (-1.74533, 1.74533),
    "elbow_flex": (-1.69, 1.69),
    "wrist_flex": (-1.65806, 1.65806),
    "wrist_roll": (-2.74385, 2.84121),
}

#: The jaw. Not an arm DOF (it moves the tool frame by exactly 0.0 m), so it is
#: kept separate to stop it being swept into a 5-vector by accident.
GRIPPER_RAD: Final[tuple[float, float]] = (-0.17453, 1.74533)

#: Canonical joint order. Every 5-vector in this package follows it.
JOINTS: Final[tuple[str, ...]] = tuple(CTRLRANGE_RAD)

#: Same limits in degrees. COMPUTED from the radians above, never typed.
LIMITS_DEG: Final[dict[str, tuple[float, float]]] = {
    name: (math.degrees(lo), math.degrees(hi)) for name, (lo, hi) in CTRLRANGE_RAD.items()
}


def clamp_deg(joint: str, value: float) -> float:
    """Clamp one joint angle (degrees) into its model-derived range."""
    lo, hi = LIMITS_DEG[joint]
    return lo if value < lo else hi if value > hi else value


def within_limits_deg(angles: dict[str, float], tol_deg: float = 0.0) -> bool:
    """True when every named joint is inside its range, within `tol_deg`."""
    return all(
        LIMITS_DEG[j][0] - tol_deg <= angles[j] <= LIMITS_DEG[j][1] + tol_deg for j in angles
    )


def span_deg(joint: str) -> float:
    """Total travel of one joint, in degrees."""
    lo, hi = LIMITS_DEG[joint]
    return hi - lo


def gripper_rad(openness: float) -> float:
    """Normalized jaw openness [0,1] -> joint angle in radians.

    The boundary between intent and actuator units. `retarget.py` speaks only in
    openness; this is the single place that knows what the servo wants.

    Measured: driving this joint from 0 to 1.5 rad moves the tool frame by
    exactly 0.000e+00 m. The jaw is fully decoupled from the arm, so no gripper
    command can perturb the IK or the arm's motion.
    """
    lo, hi = GRIPPER_RAD
    clamped = 0.0 if openness < 0.0 else 1.0 if openness > 1.0 else openness
    return lo + clamped * (hi - lo)


def gripper_openness(radians: float) -> float:
    """Inverse of `gripper_rad`, for reading the arm's actual jaw back."""
    lo, hi = GRIPPER_RAD
    return (radians - lo) / (hi - lo)
