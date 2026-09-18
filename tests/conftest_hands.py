"""Synthetic hands with exactly known geometry.

A real hand from a photo is good for "does this work". A hand you built yourself
is the only way to ask "is the answer the RIGHT one", because you know what it
should be before you run the code.
"""

from __future__ import annotations

import numpy as np

from robo_mimic.landmarks import (
    INDEX_MCP,
    INDEX_TIP,
    MIDDLE_MCP,
    N_LANDMARKS,
    PINKY_MCP,
    RING_MCP,
    THUMB_TIP,
    WRIST,
    HandLandmarks,
)

#: A flat right hand, palm facing +z, fingers along +y, thumb side at -x.
#: Chosen so the hand frame comes out as EXACTLY the identity matrix:
#:     forward = wrist -> middle knuckle   = +y
#:     across  = index -> pinky knuckle    = +x
#:     z = -(forward x across) = +z,  x = y x z = +x
PALM_HALF = 0.05
KNUCKLE_HALF = 0.02


def canonical_world() -> np.ndarray:
    """World landmarks for the canonical right hand, in metres."""
    world = np.zeros((N_LANDMARKS, 3))
    world[WRIST] = (0.0, -PALM_HALF, 0.0)
    world[MIDDLE_MCP] = (0.0, PALM_HALF, 0.0)
    world[INDEX_MCP] = (-KNUCKLE_HALF, PALM_HALF, 0.0)
    world[PINKY_MCP] = (KNUCKLE_HALF, PALM_HALF, 0.0)
    world[RING_MCP] = (0.5 * KNUCKLE_HALF, PALM_HALF, 0.0)
    world[THUMB_TIP] = (-2 * KNUCKLE_HALF, 0.0, 0.0)
    world[INDEX_TIP] = (-KNUCKLE_HALF, 3 * PALM_HALF, 0.0)
    return world


def make_hand(
    *,
    rotation: np.ndarray | None = None,
    pinch_ratio: float = 1.0,
    image_centre: tuple[float, float] = (0.5, 0.5),
    image_span: float = 0.2,
    handedness: str = "Right",
    score: float = 0.99,
    timestamp_ms: int = 0,
) -> HandLandmarks:
    """A hand with the geometry you ask for.

    `pinch_ratio` is set by placing the thumb tip the right distance from the
    index tip; palm span is fixed at 2 * PALM_HALF, so the ratio is exact.
    `image_*` place the hand on screen, which is what drives `image_position`.
    """
    world = canonical_world()
    if handedness == "Left":
        world = world @ np.diag([-1.0, 1.0, 1.0])
    if rotation is not None:
        world = world @ rotation.T

    # Distances are rotation-invariant, so placing the thumb a fixed distance
    # from the index tip sets `pinch_ratio` exactly, whatever the hand's pose.
    span = 2 * PALM_HALF
    world[THUMB_TIP] = world[INDEX_TIP] + np.array([pinch_ratio * span, 0.0, 0.0])

    # Image landmarks. Only six points matter: the four MCPs fix the palm
    # centroid, and wrist -> middle knuckle fixes the apparent span. Placing all
    # four MCPs on the same point makes the centroid exactly `image_centre`.
    image = np.zeros((N_LANDMARKS, 3))
    cx, cy = image_centre
    for index in (INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP):
        image[index] = (cx, cy, 0.0)
    image[WRIST] = (cx, cy + image_span, 0.0)

    return HandLandmarks(
        image=image,
        world=world,
        handedness=handedness,
        score=score,
        timestamp_ms=timestamp_ms,
    )


def rodrigues(axis: tuple[float, float, float] | np.ndarray, angle: float) -> np.ndarray:
    """Rotation about a unit axis. Always a valid rotation, so fuzzing never wastes a draw."""
    unit = np.asarray(axis, float)
    unit = unit / np.linalg.norm(unit)
    hat = np.array(
        [[0, -unit[2], unit[1]], [unit[2], 0, -unit[0]], [-unit[1], unit[0], 0]], float
    )
    return np.eye(3) + np.sin(angle) * hat + (1 - np.cos(angle)) * (hat @ hat)
