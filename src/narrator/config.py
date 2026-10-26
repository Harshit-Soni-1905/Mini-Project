"""Configuration and constants for Smart Obstacle Narrator."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


# Project root is two levels above src/narrator/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Default model — Ultralytics downloads it automatically on first use.
DEFAULT_MODEL = "yolo11n.pt"

# COCO classes considered relevant obstacles for a mobility-assistance
# scenario.  This is a starting set; it should be tuned based on
# evaluation in real environments.
OBSTACLE_CLASSES: set[str] = {
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "traffic light",
    "stop sign",
    "bench",
    "dog",
    "cat",
    "chair",
    "potted plant",
    "fire hydrant",
    "suitcase",
    "backpack",
    "bottle",
    "cup",
    "handbag",
    "cell phone",
    "laptop",
    "book",
    "dining table",
}

# Assistive Obstacle Taxonomy categories
WALKING_HAZARD_CLASSES: set[str] = {
    "chair",
    "couch",
    "bed",
    "dining table",
    "toilet",
    "suitcase",
    "backpack",
    "handbag",
    "bench",
    "potted plant",
    "fire hydrant",
    "stop sign",
}

DYNAMIC_HAZARD_CLASSES: set[str] = {
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "dog",
    "cat",
    "horse",
    "train",
}

NAVIGATION_LANDMARKS: set[str] = {
    "traffic light",
    "stop sign",
    "bench",
    "fire hydrant",
}

SECONDARY_OBJECT_CLASSES: set[str] = {
    "bottle",
    "cup",
    "handbag",
    "backpack",
    "cell phone",
    "laptop",
    "book",
}

# Predefined speech profiles
SPEECH_PROFILES: dict[str, set[str]] = {
    "navigation_only": WALKING_HAZARD_CLASSES | DYNAMIC_HAZARD_CLASSES | NAVIGATION_LANDMARKS,
    "full_awareness": (
        WALKING_HAZARD_CLASSES
        | DYNAMIC_HAZARD_CLASSES
        | NAVIGATION_LANDMARKS
        | SECONDARY_OBJECT_CLASSES
    ),
    "indoor": {
        "chair",
        "couch",
        "bed",
        "dining table",
        "suitcase",
        "backpack",
        "handbag",
        "potted plant",
        "person",
        "dog",
        "cat",
        "bottle",
        "cup",
        "laptop",
        "cell phone",
        "book",
    },
    "outdoor": {
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "bus",
        "truck",
        "traffic light",
        "stop sign",
        "bench",
        "dog",
        "fire hydrant",
        "backpack",
        "handbag",
        "suitcase",
    },
}


@dataclass
class DetectorConfig:
    """Settings for the YOLO detector."""

    model_path: str = DEFAULT_MODEL
    confidence_threshold: float = 0.35
    image_size: int = 640
    device: str | None = None  # None = auto-select (CUDA if available)
    obstacle_classes: set[str] = field(default_factory=lambda: OBSTACLE_CLASSES.copy())
    tracking_enabled: bool = True  # Enable ByteTrack persistent tracking
    tracker_type: str = "bytetrack.yaml"  # Tracker config (bytetrack.yaml, botsort.yaml)
    debug_detections: bool = False  # Enable detailed detection diagnostics

    def resolve_device(self) -> str:
        """Return the device string, auto-selecting CUDA when available."""
        if self.device is not None:
            return self.device
        try:
            import torch
            return "0" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"


@dataclass
class SpeakerConfig:
    """Settings for the text-to-speech engine."""

    rate: int = 175  # words per minute
    cooldown_seconds: float = 2.0  # minimum gap between repeated messages
    enabled: bool = True  # set False to mute narration entirely


@dataclass
class WalkingCorridorConfig:
    """Settings for ground-path walking corridor hazard estimation."""

    enabled: bool = True
    corridor_x_min: float = 0.25  # left boundary of walking corridor (normalized 0..1)
    corridor_x_max: float = 0.75  # right boundary of walking corridor (normalized 0..1)
    ground_y_min: float = 0.45    # vertical threshold: bottom of bounding box must be >= this to touch ground
    path_urgency_bonus: float = 25.0  # urgency boost for objects in the walking corridor


@dataclass
class EventsConfig:
    """Settings for event-based narration."""

    global_cooldown_seconds: float = 1.5  # minimum gap between any two spoken events
    track_cooldown_seconds: float = 4.0  # minimum gap before same track/proximity repeats
    announce_new_objects: bool = True  # announce when a track is first confirmed
    announce_proximity_changes: bool = True  # announce proximity escalations (FAR->MED, MED->NEAR)
    min_frames_to_confirm: int = 2  # frames a track must persist before announcing
    speech_profile: str = "full_awareness"  # "navigation_only", "full_awareness", "indoor", "outdoor"
    alert_classes: set[str] | None = None  # None = resolve via speech_profile

    def resolve_alert_classes(self) -> set[str] | None:
        """Resolve speech-eligible classes from alert_classes or active speech profile."""
        if self.alert_classes is not None:
            return self.alert_classes
        if self.speech_profile in SPEECH_PROFILES:
            return SPEECH_PROFILES[self.speech_profile]
        return None


@dataclass
class CameraConfig:
    """Settings for the live camera feed."""

    index: int = 0  # OpenCV camera index
    frame_width: int | None = None  # None = use camera default
    frame_height: int | None = None


@dataclass
class DisplayConfig:
    """Settings for the on-screen display."""

    show_boxes: bool = True
    show_fps: bool = True
    show_labels: bool = True
    window_name: str = "Smart Obstacle Narrator"


@dataclass
class DepthConfig:
    """Settings for relative depth estimation."""

    enabled: bool = True
    estimator_type: str = "geometric"  # "geometric", "none"
    horizon_y: float = 0.35  # normalized horizon y-coordinate (0.0 to 1.0)
    immediate_threshold: float = 0.75  # normalized closeness threshold
    near_threshold: float = 0.50  # normalized closeness threshold
    medium_threshold: float = 0.25  # normalized closeness threshold
    inner_crop: float = 0.5  # central crop fraction to avoid background bleeding


@dataclass
class GroundRegionConfig:
    """Settings for visible ground-region estimation (Phase 2C.1)."""

    enabled: bool = True
    estimator_type: str = "perspective"  # "perspective", "none"
    horizon_y: float = 0.35  # normalized physical horizon
    ground_y_min: float = 0.45  # vertical threshold: y >= this touches ground plane
    corridor_top_left: float = 0.35  # normalized corridor left at ground_y_min
    corridor_top_right: float = 0.65  # normalized corridor right at ground_y_min
    corridor_bottom_left: float = 0.15  # normalized corridor left at frame bottom
    corridor_bottom_right: float = 0.85  # normalized corridor right at frame bottom
    account_for_obstacles: bool = True  # subtract detected obstacle footprints from ground
    generate_mask: bool = True  # compute pixel-level uint8 ground mask


@dataclass
class HealthConfig:
    """Settings for system health monitoring and alerts."""

    enabled: bool = True
    warning_cooldown_seconds: float = 30.0  # minimum gap between repeat warnings
    stale_frame_threshold_seconds: float = 2.0  # seconds without frames before marking stale
    consecutive_capture_failures_to_alert: int = 5  # consecutive frame read failures to trigger camera alert
    max_frame_processing_delay_ms: float = 200.0  # diagnostic latency threshold
    max_inference_delay_ms: float = 100.0  # diagnostic latency threshold


@dataclass
class AppConfig:
    """Top-level application configuration."""

    detector: DetectorConfig = field(default_factory=DetectorConfig)
    speaker: SpeakerConfig = field(default_factory=SpeakerConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    events: EventsConfig = field(default_factory=EventsConfig)
    corridor: WalkingCorridorConfig = field(default_factory=WalkingCorridorConfig)
    health: HealthConfig = field(default_factory=HealthConfig)
    depth: DepthConfig = field(default_factory=DepthConfig)
    ground: GroundRegionConfig = field(default_factory=GroundRegionConfig)
    debug_detections: bool = False  # Enable detailed detection diagnostics

    @classmethod
    def from_env(cls) -> AppConfig:
        """Build config with environment-variable overrides."""
        config = cls()
        if env_conf := os.environ.get("NARRATOR_CONFIDENCE"):
            config.detector.confidence_threshold = float(env_conf)
        if env_device := os.environ.get("NARRATOR_DEVICE"):
            config.detector.device = env_device
        if env_model := os.environ.get("NARRATOR_MODEL"):
            config.detector.model_path = env_model
        if env_camera := os.environ.get("NARRATOR_CAMERA"):
            config.camera.index = int(env_camera)
        if env_mute := os.environ.get("NARRATOR_MUTE"):
            config.speaker.enabled = env_mute.lower() not in ("1", "true", "yes")
        if env_track := os.environ.get("NARRATOR_TRACK"):
            config.detector.tracking_enabled = env_track.lower() in ("1", "true", "yes")
        if env_tracker := os.environ.get("NARRATOR_TRACKER"):
            config.detector.tracker_type = env_tracker
        if env_profile := os.environ.get("NARRATOR_SPEECH_PROFILE"):
            config.events.speech_profile = env_profile
        if env_debug := os.environ.get("NARRATOR_DEBUG_DETECTIONS"):
            config.debug_detections = env_debug.lower() in ("1", "true", "yes")
        if env_health := os.environ.get("NARRATOR_HEALTH_ENABLED"):
            config.health.enabled = env_health.lower() in ("1", "true", "yes")
        if env_depth := os.environ.get("NARRATOR_DEPTH_ENABLED"):
            config.depth.enabled = env_depth.lower() in ("1", "true", "yes")
        if env_horizon := os.environ.get("NARRATOR_HORIZON_Y"):
            config.depth.horizon_y = float(env_horizon)
        if env_ground := os.environ.get("NARRATOR_GROUND_ENABLED"):
            config.ground.enabled = env_ground.lower() in ("1", "true", "yes")
        if env_ground_type := os.environ.get("NARRATOR_GROUND_ESTIMATOR"):
            config.ground.estimator_type = env_ground_type
        if env_ground_obs := os.environ.get("NARRATOR_GROUND_ACCOUNT_OBSTACLES"):
            config.ground.account_for_obstacles = env_ground_obs.lower() in ("1", "true", "yes")
        return config
