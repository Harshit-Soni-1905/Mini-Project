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
}


@dataclass
class DetectorConfig:
    """Settings for the YOLO detector."""

    model_path: str = DEFAULT_MODEL
    confidence_threshold: float = 0.35
    image_size: int = 640
    device: str | None = None  # None = auto-select (CUDA if available)
    obstacle_classes: set[str] = field(default_factory=lambda: OBSTACLE_CLASSES.copy())

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


@dataclass
class AppConfig:
    """Top-level application configuration."""

    detector: DetectorConfig = field(default_factory=DetectorConfig)
    speaker: SpeakerConfig = field(default_factory=SpeakerConfig)

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
        return config
