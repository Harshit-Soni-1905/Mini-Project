"""Ground-plane walking path and contact point hazard assessment.

In assistive mobility setups (chest-mounted or handheld camera angled
slightly downward), the immediate walking corridor occupies the lower-center
region of the camera frame. Objects whose physical contact point (bottom-center
of bounding box) touches this footstep zone constitute primary tripping/collision hazards.

Phase 2C.1 introduces GroundRegion, GroundRegionEstimator, and PerspectiveGroundRegionEstimator
to estimate the visible ground plane and trapezoidal walking corridor while accounting for
detected obstacle ground footprints.

SAFETY NOTICE:
Ground estimation alone must NEVER be used to declare a path "safe" or "clear".
It is an informational prior and visible surface approximation, NOT a clearance
guarantee or safety certification. It does not detect drop-offs, stairs, curbs,
cables, holes, or transparent obstacles.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from narrator.config import GroundRegionConfig
    from narrator.detector import Detection

logger = logging.getLogger(__name__)


class HazardZone(str, Enum):
    """Spatial zone classification relative to user's walking path."""

    WALKING_PATH = "walking_path"  # In immediate footstep corridor (critical tripping/collision hazard)
    PERIPHERAL = "peripheral"      # On ground level, but to the left or right of walking corridor
    ELEVATED = "elevated"          # Elevated off the ground (tabletop, hanging, overhead)


@dataclass(frozen=True)
class ContactPoint:
    """Estimated physical ground contact point of a detected object."""

    x: float  # horizontal center (pixels)
    y: float  # bottom of bounding box (pixels)
    normalized_x: float  # x / frame_width (0.0 to 1.0)
    normalized_y: float  # y / frame_height (0.0 to 1.0)
    zone: HazardZone
    is_on_ground_region: bool = True


@dataclass(frozen=True)
class GroundRegion:
    """Estimated visible ground surface representation in the camera frame.

    Attributes
    ----------
    polygon : np.ndarray
        Coordinates of the general ground boundary polygon (shape: (N, 2), float32).
        For perspective estimation, this represents the floor plane below the ground horizon.
    corridor_polygon : np.ndarray
        Coordinates of the immediate walking corridor trapezoid (shape: (4, 2), float32),
        narrower near the horizon and widening toward the user's feet.
    mask : np.ndarray | None
        Binary or 8-bit visible ground surface mask (shape: (H, W), uint8).
        255 = unobstructed ground, 0 = non-ground or obstacle footprint.
        Can be None if mask generation is disabled for minimal compute.
    confidence : float
        Epistemic confidence / certainty score in [0.0, 1.0].
        Reflects frame validity, sensor/variance sanity, and estimator trust.
    is_valid : bool
        True if the ground region was successfully estimated; False if missing,
        corrupted, empty, or disabled.
    ground_fraction : float
        Fraction of total frame area estimated as visible ground [0.0, 1.0].
    corridor_ground_fraction : float
        Fraction of the walking corridor estimated as visible and unobstructed [0.0, 1.0].
        When obstacles occlude the path, this drops proportionally.
    error_reason : str | None
        Descriptive string explaining why is_valid is False or why confidence is degraded.
    """

    polygon: np.ndarray
    corridor_polygon: np.ndarray
    mask: np.ndarray | None
    confidence: float
    is_valid: bool
    ground_fraction: float
    corridor_ground_fraction: float
    error_reason: str | None = None

    def is_in_ground(self, x: float, y: float) -> bool:
        """Check if pixel coordinate (x, y) is inside the estimated ground region."""
        if not self.is_valid or self.polygon is None or len(self.polygon) < 3:
            return False
        pts = np.asarray(self.polygon, dtype=np.float32)
        return cv2.pointPolygonTest(pts, (float(x), float(y)), False) >= 0

    def is_in_corridor(self, x: float, y: float) -> bool:
        """Check if pixel coordinate (x, y) is inside the walking corridor trapezoid."""
        if not self.is_valid or self.corridor_polygon is None or len(self.corridor_polygon) < 3:
            return False
        pts = np.asarray(self.corridor_polygon, dtype=np.float32)
        return cv2.pointPolygonTest(pts, (float(x), float(y)), False) >= 0

    @property
    def corridor_occlusion_fraction(self) -> float:
        """Fraction of the walking corridor occluded by detected obstacles."""
        return max(0.0, min(1.0, 1.0 - self.corridor_ground_fraction))


class GroundRegionEstimator(ABC):
    """Abstract interface for visible ground-region estimators."""

    @abstractmethod
    def estimate_ground_region(
        self,
        frame: np.ndarray | None,
        detections: list[Detection] | None = None,
    ) -> GroundRegion:
        """Estimate the visible ground surface in the frame.

        Parameters
        ----------
        frame : np.ndarray | None
            Input camera frame (H, W, 3) or (H, W).
        detections : list[Detection] | None
            Optional detected obstacles to account for ground footprint occlusion.

        Returns
        -------
        GroundRegion
            Estimated ground region with geometry, mask, confidence, and metrics.
        """
        ...


class PerspectiveGroundRegionEstimator(GroundRegionEstimator):
    """Estimates visible ground region using camera perspective geometry and obstacle masking.

    Models a chest-mounted or neck-mounted downward-looking camera geometry:
    - Horizon / ground minimum y: vertical boundary above which is non-ground / elevated.
    - Ground polygon: spans full width [0, W] from ground_y_min to frame bottom H.
    - Walking corridor trapezoid:
        - Top width (at ground_y_min): corridor_top_left to corridor_top_right (e.g. 0.35 to 0.65).
        - Bottom width (at frame bottom): corridor_bottom_left to corridor_bottom_right (e.g. 0.15 to 0.85).
    - Obstacle footprint masking:
        - When detections are provided and account_for_obstacles is True, the ground-contact
          bounding box of each obstacle is subtracted from the ground mask, reflecting visible
          unobstructed ground.
    - Fast, deterministic, sub-millisecond execution, 0 MB GPU memory overhead.
    """

    def __init__(
        self,
        *,
        horizon_y: float = 0.35,
        ground_y_min: float = 0.45,
        corridor_top_left: float = 0.35,
        corridor_top_right: float = 0.65,
        corridor_bottom_left: float = 0.15,
        corridor_bottom_right: float = 0.85,
        account_for_obstacles: bool = True,
        generate_mask: bool = True,
        base_confidence: float = 0.80,
    ) -> None:
        self.horizon_y = float(horizon_y)
        self.ground_y_min = float(ground_y_min)
        self.corridor_top_left = float(corridor_top_left)
        self.corridor_top_right = float(corridor_top_right)
        self.corridor_bottom_left = float(corridor_bottom_left)
        self.corridor_bottom_right = float(corridor_bottom_right)
        self.account_for_obstacles = bool(account_for_obstacles)
        self.generate_mask = bool(generate_mask)
        self.base_confidence = float(base_confidence)

    def estimate_ground_region(
        self,
        frame: np.ndarray | None,
        detections: list[Detection] | None = None,
    ) -> GroundRegion:
        """Estimate ground region with perspective geometry and obstacle occlusion masking."""
        # 1. Missing or None frame
        if frame is None:
            return GroundRegion(
                polygon=np.empty((0, 2), dtype=np.float32),
                corridor_polygon=np.empty((0, 2), dtype=np.float32),
                mask=None,
                confidence=0.0,
                is_valid=False,
                ground_fraction=0.0,
                corridor_ground_fraction=0.0,
                error_reason="missing_frame",
            )

        # 2. Invalid shape or empty array
        if not isinstance(frame, np.ndarray) or frame.size == 0 or len(frame.shape) < 2:
            return GroundRegion(
                polygon=np.empty((0, 2), dtype=np.float32),
                corridor_polygon=np.empty((0, 2), dtype=np.float32),
                mask=None,
                confidence=0.0,
                is_valid=False,
                ground_fraction=0.0,
                corridor_ground_fraction=0.0,
                error_reason="empty_or_invalid_frame",
            )

        h, w = frame.shape[:2]
        if h < 10 or w < 10:
            return GroundRegion(
                polygon=np.empty((0, 2), dtype=np.float32),
                corridor_polygon=np.empty((0, 2), dtype=np.float32),
                mask=None,
                confidence=0.0,
                is_valid=False,
                ground_fraction=0.0,
                corridor_ground_fraction=0.0,
                error_reason="frame_dimensions_too_small",
            )

        try:
            # 3. Confidence assessment (detect degenerate / zero variance / covered frames)
            # Use fast spatial striding with cv2.meanStdDev to check mean & variance in ~0.02ms
            confidence = self.base_confidence
            error_reason = None
            sample = frame[::8, ::8, 0] if len(frame.shape) == 3 else frame[::8, ::8]
            mean_val, std_dev = cv2.meanStdDev(sample)
            frame_mean = float(mean_val[0][0])
            frame_std = float(std_dev[0][0])

            # Degenerate frame detection:
            # 1. Low variance / flat synthetic image (std < 2.0)
            # 2. Covered camera lens or pitch black darkness (mean < 10.0, even with CMOS noise)
            # 3. Saturated / overexposed glare (mean > 248.0 with std < 5.0)
            if frame_std < 2.0:
                confidence = 0.15
                error_reason = "low_variance_degenerate_frame"
            elif frame_mean < 10.0:
                confidence = 0.10
                error_reason = "covered_lens_or_pitch_black"
            elif frame_mean > 248.0 and frame_std < 5.0:
                confidence = 0.10
                error_reason = "saturated_overexposed_frame"

            # 4. Perspective polygons
            # Ground plane: from ground_y_min to frame bottom across full width
            y_start = float(self.ground_y_min * h)
            ground_poly = np.array(
                [
                    [0.0, y_start],
                    [float(w), y_start],
                    [float(w), float(h)],
                    [0.0, float(h)],
                ],
                dtype=np.float32,
            )

            # Corridor trapezoid: narrow at top (ground_y_min), wide at bottom
            corridor_poly = np.array(
                [
                    [float(self.corridor_top_left * w), y_start],
                    [float(self.corridor_top_right * w), y_start],
                    [float(self.corridor_bottom_right * w), float(h)],
                    [float(self.corridor_bottom_left * w), float(h)],
                ],
                dtype=np.float32,
            )

            # 5. Mask and fractions computation
            if self.generate_mask:
                ground_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(ground_mask, [np.int32(ground_poly)], 255)

                corridor_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(corridor_mask, [np.int32(corridor_poly)], 255)

                total_corridor_pixels = int(cv2.countNonZero(corridor_mask))
                total_frame_pixels = float(h * w)

                visible_mask = ground_mask.copy()
                if self.account_for_obstacles and detections:
                    for det in detections:
                        x1 = max(0, int(det.x1))
                        x2 = min(w, int(det.x2))
                        y2 = min(h, int(det.y2))
                        # Object footprint touching the ground
                        y1 = max(int(y_start), int(det.y1))
                        if x2 > x1 and y2 > y1:
                            visible_mask[y1:y2, x1:x2] = 0

                visible_ground_pixels = int(cv2.countNonZero(visible_mask))
                ground_fraction = visible_ground_pixels / total_frame_pixels

                visible_corridor_pixels = int(
                    cv2.countNonZero(cv2.bitwise_and(visible_mask, corridor_mask))
                )
                corridor_ground_fraction = (
                    visible_corridor_pixels / float(max(1, total_corridor_pixels))
                )
                final_mask = visible_mask
            else:
                final_mask = None
                ground_fraction = float(1.0 - self.ground_y_min)
                corridor_ground_fraction = 1.0

            return GroundRegion(
                polygon=ground_poly,
                corridor_polygon=corridor_poly,
                mask=final_mask,
                confidence=confidence,
                is_valid=True,
                ground_fraction=ground_fraction,
                corridor_ground_fraction=corridor_ground_fraction,
                error_reason=error_reason,
            )
        except Exception as exc:
            logger.debug("Ground region estimation failed: %s", exc)
            return GroundRegion(
                polygon=np.empty((0, 2), dtype=np.float32),
                corridor_polygon=np.empty((0, 2), dtype=np.float32),
                mask=None,
                confidence=0.0,
                is_valid=False,
                ground_fraction=0.0,
                corridor_ground_fraction=0.0,
                error_reason=f"estimator_exception: {exc}",
            )


class DisabledGroundRegionEstimator(GroundRegionEstimator):
    """Fallback estimator when ground estimation is disabled or unavailable."""

    def estimate_ground_region(
        self,
        frame: np.ndarray | None,
        detections: list[Detection] | None = None,
    ) -> GroundRegion:
        return GroundRegion(
            polygon=np.empty((0, 2), dtype=np.float32),
            corridor_polygon=np.empty((0, 2), dtype=np.float32),
            mask=None,
            confidence=0.0,
            is_valid=False,
            ground_fraction=0.0,
            corridor_ground_fraction=0.0,
            error_reason="ground_estimation_disabled",
        )


def create_ground_estimator(config: GroundRegionConfig | None = None) -> GroundRegionEstimator:
    """Factory to instantiate the configured ground-region estimator."""
    if config is None or not config.enabled or config.estimator_type == "none":
        return DisabledGroundRegionEstimator()
    if config.estimator_type == "perspective":
        return PerspectiveGroundRegionEstimator(
            horizon_y=config.horizon_y,
            ground_y_min=config.ground_y_min,
            corridor_top_left=config.corridor_top_left,
            corridor_top_right=config.corridor_top_right,
            corridor_bottom_left=config.corridor_bottom_left,
            corridor_bottom_right=config.corridor_bottom_right,
            account_for_obstacles=config.account_for_obstacles,
            generate_mask=config.generate_mask,
        )
    logger.warning("Unknown ground estimator type '%s'; disabling", config.estimator_type)
    return DisabledGroundRegionEstimator()


def estimate_contact_point(
    detection: Detection,
    frame_width: int,
    frame_height: int,
    *,
    corridor_x_min: float = 0.25,
    corridor_x_max: float = 0.75,
    ground_y_min: float = 0.45,
    corridor_hysteresis: float = 0.02,
    previous_zone: HazardZone | None = None,
    ground_region: GroundRegion | None = None,
) -> ContactPoint:
    """Estimate the object's ground contact point and classify its hazard zone.

    Parameters
    ----------
    detection : Detection
        Object bounding box.
    frame_width : int
        Width of camera frame.
    frame_height : int
        Height of camera frame.
    corridor_x_min : float
        Left normalized boundary of the footstep corridor (default 0.25).
    corridor_x_max : float
        Right normalized boundary of the footstep corridor (default 0.75).
    ground_y_min : float
        Vertical threshold; y2/height must be >= this for ground contact (default 0.45).
    corridor_hysteresis : float
        Hysteresis margin around corridor boundaries to prevent zone flapping (default 0.02).
    previous_zone : HazardZone | None
        Previous frame's zone for hysteresis. If None, no hysteresis applied (first frame).
    ground_region : GroundRegion | None
        Optional estimated GroundRegion. If valid, uses the trapezoidal corridor projection
        at the contact point's height instead of a static rectangular boundary.

    Returns
    -------
    ContactPoint
        Classified contact point and zone.
    """
    if frame_width <= 0 or frame_height <= 0:
        return ContactPoint(
            x=detection.cx,
            y=detection.y2,
            normalized_x=0.5,
            normalized_y=0.5,
            zone=HazardZone.PERIPHERAL,
            is_on_ground_region=False,
        )

    cx = detection.cx
    y_bottom = detection.y2
    norm_x = cx / frame_width
    norm_y = y_bottom / frame_height

    # If the bottom of the bounding box is in the upper portion of the frame,
    # it is elevated off the ground (tabletop object, shelf, hanging sign).
    if norm_y < ground_y_min:
        zone = HazardZone.ELEVATED
        is_on_ground = False
    else:
        # Determine corridor boundaries at the contact point's height
        if (
            ground_region is not None
            and ground_region.is_valid
            and len(ground_region.corridor_polygon) >= 4
        ):
            # Trapezoidal corridor projection:
            # P0: top-left, P1: top-right, P2: bottom-right, P3: bottom-left
            p0 = ground_region.corridor_polygon[0]
            p1 = ground_region.corridor_polygon[1]
            p2 = ground_region.corridor_polygon[2]
            p3 = ground_region.corridor_polygon[3]

            y_top = float(p0[1])
            y_bot = float(p3[1])
            denom = max(1.0, y_bot - y_top)
            t = max(0.0, min(1.0, (float(y_bottom) - y_top) / denom))

            x_left = float(p0[0] + t * (p3[0] - p0[0]))
            x_right = float(p1[0] + t * (p2[0] - p1[0]))
            base_left = x_left / float(frame_width)
            base_right = x_right / float(frame_width)
            is_on_ground = ground_region.is_in_ground(cx, y_bottom)
        else:
            base_left = corridor_x_min
            base_right = corridor_x_max
            is_on_ground = True

        # Apply hysteresis around corridor boundaries to prevent flapping
        effective_left = base_left
        effective_right = base_right
        if previous_zone is not None:
            if previous_zone == HazardZone.WALKING_PATH:
                # To leave walking path, must cross boundary + hysteresis
                effective_left = base_left - corridor_hysteresis
                effective_right = base_right + corridor_hysteresis
            elif previous_zone == HazardZone.PERIPHERAL:
                # To enter walking path from peripheral, must cross boundary - hysteresis
                effective_left = base_left + corridor_hysteresis
                effective_right = base_right - corridor_hysteresis

        if effective_left <= norm_x <= effective_right:
            zone = HazardZone.WALKING_PATH
        else:
            zone = HazardZone.PERIPHERAL

    return ContactPoint(
        x=cx,
        y=y_bottom,
        normalized_x=norm_x,
        normalized_y=norm_y,
        zone=zone,
        is_on_ground_region=is_on_ground,
    )
