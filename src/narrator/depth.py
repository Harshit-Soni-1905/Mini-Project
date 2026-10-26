"""Monocular and geometric depth estimation abstractions.

Provides a sensor-agnostic DepthEstimator interface, camera perspective
geometric estimator, neural monocular depth estimator skeleton, and relative
distance classification into Immediate, Near, Medium, and Far categories.

HONEST DEPTH SEMANTICS & LIMITATIONS DISCLAIMER:
- Standard monocular webcam RGB images alone CANNOT provide reliable absolute
  physical metric distance in metres.
- The estimates produced by GeometricDepthEstimator are strictly RELATIVE
  closeness scores normalized to [0.0, 1.0], derived from ground-plane
  camera perspective (lower in frame = closer to user's feet).
- A relative depth score must NEVER be reported as calibrated physical metres.
- Metric distance measurement requires physical calibration (camera intrinsics,
  mounting height, tilt angle, ground plane fitting) or dedicated depth
  hardware (stereo baseline, ToF, LiDAR).
- Existing bounding-box height heuristic is preserved as an automatic fallback
  whenever depth maps are missing, invalid, or raise errors.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

from narrator.distance import Proximity

if TYPE_CHECKING:
    from narrator.detector import Detection

logger = logging.getLogger(__name__)


class DepthCategory(str, Enum):
    """Relative distance zones tailored for assistive mobility.

    Note: These represent qualitative closeness zones along the camera's
    ground-plane perspective field, NOT calibrated metric metres.
    """

    IMMEDIATE = "immediate"  # Highest closeness / immediate reaction zone
    NEAR = "near"            # Close proximity / active hazard zone
    MEDIUM = "medium"        # Intermediate distance / approaching zone
    FAR = "far"              # Distant background / peripheral awareness


def depth_to_proximity(category: DepthCategory | None) -> Proximity | None:
    """Map a DepthCategory to the existing Proximity enum for narration consistency."""
    if category is None:
        return None
    if category in (DepthCategory.IMMEDIATE, DepthCategory.NEAR):
        return Proximity.NEAR
    elif category == DepthCategory.MEDIUM:
        return Proximity.MEDIUM
    elif category == DepthCategory.FAR:
        return Proximity.FAR
    return None


class DepthEstimator(ABC):
    """Abstract base class for depth estimation.

    Can be backed by geometric perspective heuristics, monocular neural networks
    (Depth Anything V2, MiDaS), or hardware stereo/RGB-D sensors (RealSense, OAK-D).
    """

    @property
    @abstractmethod
    def is_inverse_depth(self) -> bool:
        """True if higher values indicate closer distance (e.g. disparity/closeness maps)."""
        pass

    @property
    def is_available(self) -> bool:
        """True if the depth estimator is ready and functional."""
        return True

    @abstractmethod
    def estimate_depth(self, frame: np.ndarray) -> np.ndarray:
        """Estimate relative depth for an RGB/BGR frame.

        Parameters
        ----------
        frame : np.ndarray
            Input BGR frame (H x W x 3).

        Returns
        -------
        np.ndarray
            Float32 depth map normalized to [0.0, 1.0] with shape (H, W).
        """
        pass

    def get_obstacle_depth(
        self,
        depth_map: np.ndarray | None,
        detection: Detection,
        *,
        inner_crop: float = 0.5,
    ) -> float:
        """Sample representative depth for a detection.

        Samples the central inner crop of the bounding box to avoid
        background depth bleeding around obstacle boundaries.

        Parameters
        ----------
        depth_map : np.ndarray | None
            2D float array (H, W).
        detection : Detection
            Bounding box coordinates.
        inner_crop : float
            Fraction of the box width and height to sample from center (default 0.5).

        Returns
        -------
        float
            Median depth value in [0.0, 1.0].
        """
        if depth_map is None or depth_map.size == 0:
            return 0.5

        h, w = depth_map.shape[:2]
        x1 = max(0, int(detection.x1))
        y1 = max(0, int(detection.y1))
        x2 = min(w, int(detection.x2))
        y2 = min(h, int(detection.y2))

        if x2 <= x1 or y2 <= y1:
            return 0.5

        # Inner crop to avoid background bleeding
        box_w = x2 - x1
        box_h = y2 - y1
        margin_x = int(box_w * (1.0 - inner_crop) / 2.0)
        margin_y = int(box_h * (1.0 - inner_crop) / 2.0)

        cx1 = min(x1 + margin_x, x2 - 1)
        cy1 = min(y1 + margin_y, y2 - 1)
        cx2 = max(x2 - margin_x, cx1 + 1)
        cy2 = max(y2 - margin_y, cy1 + 1)

        patch = depth_map[cy1:cy2, cx1:cx2]
        if patch.size == 0:
            return 0.5

        val = float(np.median(patch))
        if np.isnan(val) or np.isinf(val):
            return 0.5

        return float(np.clip(val, 0.0, 1.0))

    def classify_distance(
        self,
        sampled_depth: float | None,
        *,
        immediate_thresh: float = 0.75,
        near_thresh: float = 0.50,
        medium_thresh: float = 0.25,
    ) -> DepthCategory:
        """Convert sampled normalized depth to categorical distance.

        Parameters
        ----------
        sampled_depth : float | None
            Normalized depth value in [0.0, 1.0].
        immediate_thresh : float
            Threshold for immediate proximity.
        near_thresh : float
            Threshold for near proximity.
        medium_thresh : float
            Threshold for medium proximity.

        Returns
        -------
        DepthCategory
            Categorical distance classification.
        """
        if sampled_depth is None or np.isnan(sampled_depth):
            return DepthCategory.MEDIUM

        # If model outputs true distance (0.0 = closest, 1.0 = farthest),
        # invert so metric consistently means higher value = closer obstacle.
        closeness = sampled_depth if self.is_inverse_depth else (1.0 - sampled_depth)

        if closeness >= immediate_thresh:
            return DepthCategory.IMMEDIATE
        elif closeness >= near_thresh:
            return DepthCategory.NEAR
        elif closeness >= medium_thresh:
            return DepthCategory.MEDIUM
        else:
            return DepthCategory.FAR


class GeometricDepthEstimator(DepthEstimator):
    """Zero-GPU geometric depth estimator based on ground-plane perspective.

    In assistive navigation with a chest-mounted camera angled slightly downward,
    vertical position on the ground plane correlates with distance: the bottom of
    the image represents the immediate ground in front of the user's feet, while
    the horizon represents distant space.

    Zero VRAM overhead, sub-millisecond execution (< 0.2ms).
    """

    def __init__(
        self,
        horizon_y: float = 0.35,
        *,
        immediate_thresh: float = 0.75,
        near_thresh: float = 0.50,
        medium_thresh: float = 0.25,
        inner_crop: float = 0.5,
    ) -> None:
        self._horizon_y = horizon_y
        self.immediate_thresh = immediate_thresh
        self.near_thresh = near_thresh
        self.medium_thresh = medium_thresh
        self.inner_crop = inner_crop

    @property
    def is_inverse_depth(self) -> bool:
        # 1.0 = bottom of image (immediate/closest), 0.0 = horizon (farthest)
        return True

    def estimate_depth(self, frame: np.ndarray) -> np.ndarray:
        """Generate relative depth gradient map based on ground plane perspective."""
        if frame is None or frame.size == 0:
            return np.zeros((480, 640), dtype=np.float32)

        h, w = frame.shape[:2]
        if h <= 0 or w <= 0:
            return np.zeros((1, 1), dtype=np.float32)

        y_indices = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, np.newaxis]

        # Below horizon, closeness increases linearly down to the feet (bottom of frame)
        denom = 1.0 - self._horizon_y
        if denom <= 0.0:
            denom = 1e-4

        closeness = np.clip(
            (y_indices - self._horizon_y) / denom,
            0.0,
            1.0,
        )
        return np.repeat(closeness, w, axis=1)


class MonocularDepthEstimator(DepthEstimator):
    """Neural monocular depth estimator skeleton for PyTorch / ONNX models.

    Designed for lightweight models like Depth Anything V2 Small or MiDaS Small.
    When uninitialized or without downloaded weights, falls back gracefully to
    the zero-overhead GeometricDepthEstimator.
    """

    def __init__(
        self,
        model_name: str = "depth_anything_v2_vits",
        device: str = "cuda",
        *,
        use_fp16: bool = True,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.use_fp16 = use_fp16
        self._model = None
        self._is_inverse = True
        self._fallback = GeometricDepthEstimator()

    @property
    def is_inverse_depth(self) -> bool:
        return self._is_inverse

    @property
    def is_available(self) -> bool:
        return self._model is not None or self._fallback is not None

    def estimate_depth(self, frame: np.ndarray) -> np.ndarray:
        """Estimate monocular depth. Falls back to geometric estimator if model uninitialized."""
        if self._model is None:
            # Fallback to perspective geometric estimator
            return self._fallback.estimate_depth(frame)

        # Placeholder for future loaded neural weights
        return self._fallback.estimate_depth(frame)

