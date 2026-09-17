"""21 landmarks -> one rigid hand pose. Pure.

THE FRAME
    origin  palm centre = mean of the 4 MCP knuckles
    y       toward the fingers (wrist -> middle knuckle)
    z       out of the palm, i.e. the way the palm faces
    x       y cross z, completing a right-handed frame

WHY THESE POINTS
Only the knuckles are rigid. Fingertips bend, so any frame built on them
rotates when you curl a finger. Measured in Phase 1: the knuckle mean sits
0.44-0.75 cm from MediaPipe's own world origin, while the 21-point centroid is
2.5-2.7 cm away and moves with every curl.

TWO THINGS THAT BITE
1. `forward` and `across` are NOT perpendicular. Measured: dot -0.26 and -0.42,
   so about 75 and 115 degrees. Assigning them directly as axes gives a sheared
   frame. The cross product fixes this for free -- see `_orthonormal`.

2. Left and right hands MIRROR. Going index-knuckle -> pinky-knuckle runs the
   opposite way around a left palm than a right one, so `forward x across`
   points out of one palm and into the other. Measured on two palms facing the
   same way: dot -0.5687. Uncorrected, swapping hands flips the robot.
   `_PALM_NORMAL_SIGN` corrects it.
"""

from __future__ import annotations

import numpy as np

from .landmarks import F64, INDEX_MCP, MIDDLE_MCP, PALM, PINKY_MCP, WRIST, HandLandmarks
from .types import Pose

#: Sign that makes `forward x across` point OUT of the palm for both hands.
#: Right: index->pinky runs clockwise seen from outside, so the cross points in.
_PALM_NORMAL_SIGN: dict[str, float] = {"Right": -1.0, "Left": +1.0}

#: Below this, the palm triangle has collapsed and its normal is meaningless.
#: A real palm gives ~0.006 m^2-ish; 1e-6 is far below any plausible hand.
MIN_TRIANGLE = 1e-6


class DegenerateHandError(ValueError):
    """The palm triangle collapsed, so no frame can be built from it.

    Raised rather than returned because it means the landmarks are nonsense, not
    that the hand is absent. An absent hand is `Frame.best() is None`.
    """


def _orthonormal(forward: F64, across: F64, sign: float) -> F64:
    """Palm triangle -> a proper rotation matrix (columns are x, y, z).

    `forward` is kept exactly: it spans the whole palm, so it is the longest and
    least noisy baseline. Everything else is derived from it.

    No Gram-Schmidt projection step is needed. `cross(a, b)` is perpendicular to
    both by definition, so z is already perpendicular to y, and x = y cross z
    closes the frame. Orthogonality comes from the algebra, not from a fix-up.
    """
    y = forward / np.linalg.norm(forward)
    normal = sign * np.cross(forward, across)
    if float(np.linalg.norm(normal)) < MIN_TRIANGLE:
        raise DegenerateHandError("palm triangle is collinear; its normal is undefined")
    z = normal / np.linalg.norm(normal)
    x = np.cross(y, z)
    return np.column_stack([x, y, z])


def hand_pose(hand: HandLandmarks) -> Pose:
    """The hand's rigid pose, from world landmarks.

    Position is the palm centre in MediaPipe's world frame, which is ~0 by
    construction -- it is NOT where the hand is in the room. Use the orientation;
    for position see `image_position`. (ADR-002.)
    """
    world = hand.world
    forward = world[MIDDLE_MCP] - world[WRIST]
    across = world[PINKY_MCP] - world[INDEX_MCP]
    if float(np.linalg.norm(forward)) < MIN_TRIANGLE:
        raise DegenerateHandError("wrist and middle knuckle coincide")
    sign = _PALM_NORMAL_SIGN[hand.handedness]
    return Pose(
        position=world[list(PALM)].mean(axis=0),
        rotation=_orthonormal(forward, across, sign),
    )


def image_span(hand: HandLandmarks) -> float:
    """Apparent palm length in normalized image units. The depth cue.

    A hand twice as far away looks half as long, so `span ~ 1/distance`. This is
    the only distance information a single camera offers, and it is relative:
    it cannot say how far, only whether the hand got nearer or further.
    """
    delta = hand.image[MIDDLE_MCP, :2] - hand.image[WRIST, :2]
    return float(np.linalg.norm(delta))


def image_position(hand: HandLandmarks, aspect: float = 1.0) -> F64:
    """Hand position in PALM-SPAN units. The differential signal for ADR-002.

    The camera gives a bearing, not a position: a lateral move of `X` at
    distance `d` shifts the image by `X/d`. Apparent span `s` goes as `1/d`.
    So dividing image offset by span cancels the distance:

        (X/d) / (1/d) = X          <- lateral, in palm spans
        1/s                        <- depth, in palm spans

    Both come out in units of the operator's own hand, so a big hand and a small
    hand produce the same numbers. That is what removes per-operator
    calibration. Only DIFFERENCES of this vector are meaningful; the origin is
    the top-left of the frame and means nothing.

    `aspect` is height/width. Image y is normalized by height and x by width, so
    without it a non-square frame stretches vertical motion.
    """
    span = image_span(hand)
    if span < 1e-9:
        raise DegenerateHandError("hand has zero apparent size")
    centre = hand.image[list(PALM), :2].mean(axis=0)
    return np.array([centre[0] / span, centre[1] * aspect / span, 1.0 / span], dtype=np.float64)
