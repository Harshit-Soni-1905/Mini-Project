"""Directional mapping — bounding box to image-relative direction.

Classifies detections into left / center / right zones based on the
horizontal position of the bounding box center relative to the frame width.

NOTE: This is strictly an image-relative direction within the camera's
field of view.  It does NOT represent the user's actual walking direction,
body orientation, or 3D bearing.
"""

from __future__ import annotations

from enum import Enum


class Direction(str, Enum):
    """Image-relative horizontal zone."""

    LEFT = "left"
    CENTER = "ahead"
    RIGHT = "right"


def estimate_direction(
    cx: float,
    frame_width: int,
    *,
    left_boundary: float = 0.33,
    right_boundary: float = 0.67,
) -> Direction:
    """Map a bounding-box center-x coordinate to an image-relative direction.

    Parameters
    ----------
    cx : float
        Horizontal center of the bounding box in pixels.
    frame_width : int
        Width of the camera frame in pixels.
    left_boundary : float
        Fraction of frame width where the left zone ends (default: 0.33).
    right_boundary : float
        Fraction of frame width where the right zone starts (default: 0.67).

    Returns
    -------
    Direction
        Direction.LEFT, Direction.CENTER, or Direction.RIGHT.
    """
    if frame_width <= 0:
        return Direction.CENTER

    ratio = cx / frame_width

    if ratio < left_boundary:
        return Direction.LEFT
    elif ratio > right_boundary:
        return Direction.RIGHT
    else:
        return Direction.CENTER
