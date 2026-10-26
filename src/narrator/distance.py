"""Proximity estimation from bounding-box geometry.

Categorises detections into coarse, explicitly approximate proximity bins:
near, medium, or far.

LIMITATIONS & DISCLAIMER:
- This is a crude heuristic based on bounding-box height relative to the
  frame height (larger box in frame -> assumed closer).
- It does NOT measure metres, feet, or physical distance.
- An object that is small in real life (e.g. a cat) will appear "far" even
  when close, while a large object (e.g. a bus) will appear "near" even
  when further away.
- A future stage will replace or supplement this with a monocular depth
  estimation model (e.g. Depth Anything, MiDaS).
"""

from __future__ import annotations

from enum import Enum


class Proximity(str, Enum):
    """Coarse, approximate proximity category."""

    NEAR = "near"
    MEDIUM = "medium"
    FAR = "far"


def estimate_proximity(
    box_height: float,
    frame_height: int,
    *,
    near_threshold: float = 0.50,
    medium_threshold: float = 0.20,
) -> Proximity:
    """Estimate approximate proximity from bounding-box height.

    Parameters
    ----------
    box_height : float
        Height of the bounding box in pixels.
    frame_height : int
        Height of the camera frame in pixels.
    near_threshold : float
        Fraction of frame height above which an object is considered "near".
        Default: 0.50 (box occupies more than half the frame height).
    medium_threshold : float
        Fraction of frame height above which an object is considered "medium".
        Default: 0.20 (box occupies between 20% and 50% of frame height).

    Returns
    -------
    Proximity
        Proximity.NEAR, Proximity.MEDIUM, or Proximity.FAR.
    """
    if frame_height <= 0:
        return Proximity.FAR

    ratio = box_height / frame_height

    if ratio >= near_threshold:
        return Proximity.NEAR
    elif ratio >= medium_threshold:
        return Proximity.MEDIUM
    else:
        return Proximity.FAR
