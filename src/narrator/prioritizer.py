"""Hazard prioritization — rank detections by urgency.

Ranks detected obstacles so that the most critical warning is spoken first.

Urgency considers:
1. Proximity: "near" is far more urgent than "far".
2. Position: objects directly ahead ("center") are more urgent than
   peripheral ones ("left", "right").
3. Object category weight: dynamic objects (vehicles, cyclists) receive
   a modest urgency boost over static obstacles (benches, chairs).
   NOTE: Category weight does NOT equate object class alone with danger;
   it is a secondary tie-breaker after proximity and position.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from narrator.detector import Detection
from narrator.direction import Direction, estimate_direction
from narrator.distance import Proximity, estimate_proximity
from narrator.ground_plane import ContactPoint, HazardZone, estimate_contact_point

if TYPE_CHECKING:
    import numpy as np
    from narrator.depth import DepthCategory, DepthEstimator
    from narrator.ground_plane import GroundRegion

logger = logging.getLogger(__name__)

# Urgency weights by proximity (highest priority factor)
PROXIMITY_WEIGHTS: dict[Proximity, float] = {
    Proximity.NEAR: 100.0,
    Proximity.MEDIUM: 50.0,
    Proximity.FAR: 10.0,
}

# Urgency weights by image-relative position
DIRECTION_WEIGHTS: dict[Direction, float] = {
    Direction.CENTER: 20.0,  # directly in path
    Direction.LEFT: 5.0,
    Direction.RIGHT: 5.0,
}

# Category urgency modifiers (secondary factor, 0.0 - 15.0)
CATEGORY_WEIGHTS: dict[str, float] = {
    # High-speed dynamic
    "car": 15.0,
    "truck": 15.0,
    "bus": 15.0,
    "motorcycle": 15.0,
    # Medium-speed dynamic
    "bicycle": 10.0,
    "person": 8.0,
    "dog": 8.0,
    # Static walking path obstacles & fixtures (tripping/collision hazards)
    "chair": 6.0,
    "dining table": 6.0,
    "couch": 6.0,
    "bed": 6.0,
    "bench": 5.0,
    "potted plant": 4.0,
    "suitcase": 4.0,
    "fire hydrant": 4.0,
    "stop sign": 3.0,
    "traffic light": 3.0,
    # Secondary everyday objects (lower baseline urgency)
    "backpack": 2.0,
    "handbag": 1.5,
    "laptop": 1.5,
    "cell phone": 1.0,
    "bottle": 1.0,
    "cup": 1.0,
    "book": 0.5,
    "cat": 2.0,
}


@dataclass(frozen=True)
class PrioritizedObstacle:
    """A detection annotated with directional and proximity estimates, plus urgency."""

    detection: Detection
    direction: Direction
    proximity: Proximity
    urgency_score: float
    contact_point: ContactPoint | None = None
    relative_depth: float | None = None
    depth_category: DepthCategory | None = None

    @property
    def label(self) -> str:
        return self.detection.label

    @property
    def track_id(self) -> int | None:
        return self.detection.track_id

    @property
    def is_in_walking_path(self) -> bool:
        return self.contact_point is not None and self.contact_point.zone == HazardZone.WALKING_PATH

    @property
    def corridor_overlap_ratio(self) -> float:
        return self.contact_point.corridor_overlap_ratio if self.contact_point is not None else 0.0

    @property
    def ground_overlap_ratio(self) -> float:
        return self.contact_point.ground_overlap_ratio if self.contact_point is not None else 0.0

    @property
    def speech_text(self) -> str:
        """Generate a concise warning phrase for text-to-speech.

        Examples:
        - "near car ahead"
        - "bicycle left"
        - "person right"
        """
        parts: list[str] = []
        if self.proximity == Proximity.NEAR:
            parts.append("near")
        parts.append(self.label)
        parts.append(self.direction.value)
        return " ".join(parts)


def prioritize_detections(
    detections: list[Detection],
    frame_width: int,
    frame_height: int,
    *,
    left_boundary: float = 0.33,
    right_boundary: float = 0.67,
    near_threshold: float = 0.50,
    medium_threshold: float = 0.20,
    corridor_x_min: float = 0.25,
    corridor_x_max: float = 0.75,
    ground_y_min: float = 0.45,
    corridor_hysteresis: float = 0.02,
    path_urgency_bonus: float = 0.0,
    depth_map: np.ndarray | None = None,
    depth_estimator: DepthEstimator | None = None,
    previous_zones: dict[int | str | None, HazardZone] | None = None,
    ground_region: GroundRegion | None = None,
) -> list[PrioritizedObstacle]:
    """Annotate and sort detections by urgency (highest first).

    Returns an empty list if there are no detections.
    """
    if not detections:
        return []

    obstacles: list[PrioritizedObstacle] = []
    for det in detections:
        direction = estimate_direction(
            det.cx,
            frame_width,
            left_boundary=left_boundary,
            right_boundary=right_boundary,
        )

        # 1. Base proximity from bounding-box height heuristic (proven fallback)
        heuristic_proximity = estimate_proximity(
            det.height,
            frame_height,
            near_threshold=near_threshold,
            medium_threshold=medium_threshold,
        )

        # 2. Relative depth estimation if depth_map and depth_estimator are provided
        relative_depth = None
        depth_category = None
        proximity = heuristic_proximity

        if depth_map is not None and depth_estimator is not None:
            try:
                sampled = depth_estimator.get_obstacle_depth(depth_map, det)
                if sampled is not None:
                    import numpy as np
                    if not np.isnan(sampled):
                        relative_depth = float(sampled)
                        depth_category = depth_estimator.classify_distance(sampled)
                        from narrator.depth import depth_to_proximity
                        depth_prox = depth_to_proximity(depth_category)
                        if depth_prox is not None:
                            proximity = depth_prox
            except Exception as e:
                logger.debug("Depth sampling failed for detection %s: %s", det.label, e)
                relative_depth = None
                depth_category = None
                proximity = heuristic_proximity

        prev_zone = None
        if previous_zones is not None:
            key = det.track_id if det.track_id is not None else det.label
            prev_zone = previous_zones.get(key)

        contact = estimate_contact_point(
            det,
            frame_width,
            frame_height,
            corridor_x_min=corridor_x_min,
            corridor_x_max=corridor_x_max,
            ground_y_min=ground_y_min,
            corridor_hysteresis=corridor_hysteresis,
            previous_zone=prev_zone,
            ground_region=ground_region,
        )

        # Ground confidence factor: if ground_region is valid and present, use its confidence;
        # otherwise default to 1.0 so that missing/invalid ground estimation never suppresses obstacles.
        if ground_region is not None and ground_region.is_valid:
            conf_factor = max(0.1, min(1.0, float(ground_region.confidence)))
        else:
            conf_factor = 1.0

        if contact.zone == HazardZone.WALKING_PATH:
            overlap_factor = max(0.35, min(1.0, contact.corridor_overlap_ratio))
            bonus = path_urgency_bonus * conf_factor * overlap_factor
        elif contact.zone == HazardZone.PERIPHERAL and contact.corridor_overlap_ratio >= 0.10:
            flanking_cap = 0.25 * path_urgency_bonus
            bonus = min(flanking_cap, path_urgency_bonus * conf_factor * contact.corridor_overlap_ratio)
        else:
            bonus = 0.0
        depth_bonus = (relative_depth * 5.0) if relative_depth is not None else 0.0

        score = (
            PROXIMITY_WEIGHTS.get(proximity, 0.0)
            + DIRECTION_WEIGHTS.get(direction, 0.0)
            + CATEGORY_WEIGHTS.get(det.label, 1.0)
            + bonus
            + depth_bonus
            + det.confidence * 2.0  # slight confidence tie-breaker
        )

        obstacles.append(
            PrioritizedObstacle(
                detection=det,
                direction=direction,
                proximity=proximity,
                urgency_score=score,
                contact_point=contact,
                relative_depth=relative_depth,
                depth_category=depth_category,
            )
        )

    obstacles.sort(key=lambda o: o.urgency_score, reverse=True)
    return obstacles
