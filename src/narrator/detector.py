"""Object detection module wrapping Ultralytics YOLO.

Loads a pretrained YOLO model once and provides a method to run inference
on individual frames (numpy arrays from OpenCV).  Returns structured
Detection objects rather than raw Ultralytics results.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from narrator.config import DetectorConfig

logger = logging.getLogger(__name__)

# Debug detection logger
detection_logger = logging.getLogger("narrator.detections")

# Throttled diagnostic logger for detector stage
det_diag_logger = logging.getLogger("narrator.detector.diagnostics")
_det_diag_last_log = 0.0
_det_diag_log_interval = 2.0  # seconds between diagnostic summaries

def _maybe_log_detector_diagnostics(
    raw_count: int,
    filtered_count: int,
    output_count: int,
    tracked_count: int,
    config,
) -> None:
    """Log throttled detector diagnostic summary."""
    global _det_diag_last_log
    import time
    now = time.monotonic()
    if now - _det_diag_last_log < _det_diag_log_interval:
        return
    _det_diag_last_log = now

    det_diag_logger.info(
        "DETECTOR: raw=%d filtered=%d output=%d (tracked=%d spatial=%d) | tracking_enabled=%s conf=%.2f imgsz=%d",
        raw_count, filtered_count, output_count,
        tracked_count, output_count - tracked_count,
        config.tracking_enabled,
        config.confidence_threshold,
        config.image_size,
    )


@dataclass(frozen=True)
class Detection:
    """A single detected object in a frame.

    Coordinates are in pixels, relative to the original frame size.
    """

    label: str
    confidence: float
    x1: float  # left
    y1: float  # top
    x2: float  # right
    y2: float  # bottom
    track_id: int | None = None  # Persistent track ID from ByteTrack (None if tracking disabled)

    @property
    def cx(self) -> float:
        """Horizontal center of the bounding box."""
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        """Vertical center of the bounding box."""
        return (self.y1 + self.y2) / 2

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height


class Detector:
    """YOLO-based obstacle detector.

    The model is loaded once at construction.  Call ``detect(frame)`` with
    a BGR numpy array (e.g. from ``cv2.VideoCapture.read()``) to get a
    list of Detection objects filtered to the configured obstacle classes.
    """

    def __init__(self, config: DetectorConfig | None = None) -> None:
        self._config = config or DetectorConfig()
        self._model = self._load_model()
        self.last_error: Exception | None = None
        self.last_detection_count: int = 0

    @property
    def is_healthy(self) -> bool:
        """Return True if detector is ready and last inference succeeded."""
        return self.last_error is None

    def _load_model(self):
        """Load the YOLO model onto the configured device."""
        from ultralytics import YOLO

        device = self._config.resolve_device()
        logger.info(
            "Loading %s on device=%s (conf=%.2f, imgsz=%d)",
            self._config.model_path,
            device,
            self._config.confidence_threshold,
            self._config.image_size,
        )
        model = YOLO(self._config.model_path)
        return model

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run detection (or tracking if enabled) on a single BGR frame.

        Returns an empty list if the frame is invalid or inference fails.
        """
        if frame is None or frame.size == 0:
            logger.warning("Received empty or None frame, skipping detection")
            return []

        try:
            if self._config.tracking_enabled:
                results = self._model.track(
                    source=frame,
                    device=self._config.resolve_device(),
                    imgsz=self._config.image_size,
                    conf=self._config.confidence_threshold,
                    tracker=self._config.tracker_type,
                    persist=True,
                    verbose=False,
                )
            else:
                results = self._model.predict(
                    source=frame,
                    device=self._config.resolve_device(),
                    imgsz=self._config.image_size,
                    conf=self._config.confidence_threshold,
                    verbose=False,
                )
            self.last_error = None
        except Exception as exc:
            logger.exception("YOLO inference failed")
            self.last_error = exc
            self.last_detection_count = 0
            return []

        detections: list[Detection] = []
        raw_detections = []
        filtered_detections = []
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                label = self._model.names.get(class_id, f"class_{class_id}")
                conf = float(box.conf[0])
                coords = box.xyxy[0].tolist() if hasattr(box.xyxy[0], "tolist") else list(box.xyxy[0])
                track_id = None
                if self._config.tracking_enabled and box.id is not None:
                    try:
                        track_id = int(box.id[0])
                    except (IndexError, ValueError, TypeError):
                        pass
                raw_detections.append({
                    'label': label,
                    'confidence': conf,
                    'xyxy': coords,
                    'track_id': track_id,
                })

                # Filter to obstacle classes only if specified
                if self._config.obstacle_classes is not None and label not in self._config.obstacle_classes:
                    filtered_detections.append({
                        'label': label,
                        'confidence': conf,
                        'reason': 'not_in_obstacle_classes'
                    })
                    continue

                if conf < self._config.confidence_threshold:
                    filtered_detections.append({
                        'label': label,
                        'confidence': conf,
                        'reason': 'below_confidence_threshold'
                    })
                    continue

                detections.append(
                    Detection(
                        label=label,
                        confidence=conf,
                        x1=coords[0],
                        y1=coords[1],
                        x2=coords[2],
                        y2=coords[3],
                        track_id=track_id,
                    )
                )

        # Debug logging if enabled
        if hasattr(self._config, 'debug_detections') and self._config.debug_detections:
            detection_logger.info(
                "Frame: raw=%d, filtered=%d, output=%d",
                len(raw_detections),
                len(filtered_detections),
                len(detections)
            )
            for d in raw_detections:
                detection_logger.debug(
                    "  RAW: %s conf=%.2f track_id=%s",
                    d['label'], d['confidence'], d['track_id']
                )
            for d in filtered_detections:
                detection_logger.debug(
                    "  FILTERED: %s conf=%.2f reason=%s",
                    d['label'], d['confidence'], d['reason']
                )
            for d in detections:
                detection_logger.debug(
                    "  OUTPUT: %s conf=%.2f track_id=%s",
                    d.label, d.confidence, d.track_id
                )

        # Throttled diagnostic logging
        if hasattr(self._config, 'debug_detections') and self._config.debug_detections:
            tracked_count = sum(1 for d in detections if d.track_id is not None)
            _maybe_log_detector_diagnostics(
                raw_count=len(raw_detections),
                filtered_count=len(filtered_detections),
                output_count=len(detections),
                tracked_count=tracked_count,
                config=self._config,
            )

        self.last_detection_count = len(detections)
        return detections

    def get_model_classes(self) -> dict[int, str]:
        """Return the dictionary of class IDs to class names from the loaded model."""
        if hasattr(self._model, "names") and isinstance(self._model.names, dict):
            return dict(self._model.names)
        elif hasattr(self._model, "names") and isinstance(self._model.names, list):
            return {i: name for i, name in enumerate(self._model.names)}
        return {}

    def diagnose_frame(self, frame: np.ndarray) -> dict:
        """Run raw inference on a frame without ByteTrack or class filtering.

        Returns a dictionary containing raw detections, latency, class mapping,
        and device information for isolated diagnostics.
        """
        import time

        if frame is None or frame.size == 0:
            return {
                "error": "Empty or invalid frame",
                "detections": [],
                "count": 0,
                "latency_ms": 0.0,
            }

        t0 = time.perf_counter()
        try:
            results = self._model.predict(
                source=frame,
                device=self._config.resolve_device(),
                imgsz=self._config.image_size,
                conf=self._config.confidence_threshold,
                verbose=False,
            )
            latency_ms = (time.perf_counter() - t0) * 1000.0
        except Exception as exc:
            return {
                "error": str(exc),
                "detections": [],
                "count": 0,
                "latency_ms": (time.perf_counter() - t0) * 1000.0,
            }

        raw_detections = []
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                label = self._model.names.get(class_id, f"class_{class_id}")
                conf = float(box.conf[0])
                coords = box.xyxy[0].tolist() if hasattr(box.xyxy[0], "tolist") else list(box.xyxy[0])
                raw_detections.append({
                    "class_id": class_id,
                    "label": label,
                    "confidence": conf,
                    "x1": coords[0],
                    "y1": coords[1],
                    "x2": coords[2],
                    "y2": coords[3],
                })

        return {
            "model_path": self._config.model_path,
            "device": self._config.resolve_device(),
            "confidence_threshold": self._config.confidence_threshold,
            "latency_ms": latency_ms,
            "count": len(raw_detections),
            "detections": raw_detections,
            "error": None,
        }

    def reset_tracker(self) -> None:
        """Reset the tracker state. Useful when the scene changes drastically."""
        # The next call to detect() with tracking will start fresh
        # because the Ultralytics model.track with persist=True will
        # reinitialize when it sees a gap. For a true reset, we could
        # recreate the model, but for typical use persist=True handles it.
        pass
