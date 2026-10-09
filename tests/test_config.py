"""Unit tests for project configuration and basic imports.

Run with: pytest
"""

from __future__ import annotations


class TestPackageImports:
    """Verify that core dependencies import without error."""

    def test_import_narrator(self):
        import narrator
        assert hasattr(narrator, "__version__")
        assert narrator.__version__ == "0.1.0"

    def test_import_narrator_config(self):
        from narrator.config import AppConfig, DetectorConfig, SpeakerConfig
        assert AppConfig is not None
        assert DetectorConfig is not None
        assert SpeakerConfig is not None

    def test_import_torch(self):
        import torch
        assert hasattr(torch, "__version__")

    def test_import_torchvision(self):
        import torchvision
        assert hasattr(torchvision, "__version__")

    def test_import_ultralytics(self):
        from ultralytics import YOLO
        assert YOLO is not None

    def test_import_cv2(self):
        import cv2
        assert hasattr(cv2, "__version__")

    def test_import_numpy(self):
        import numpy
        assert hasattr(numpy, "__version__")

    def test_import_pyttsx3(self):
        import pyttsx3
        assert pyttsx3 is not None


class TestDetectorConfig:
    """Verify DetectorConfig defaults and methods."""

    def test_defaults(self):
        from narrator.config import DetectorConfig
        cfg = DetectorConfig()
        assert cfg.model_path == "yolo11n.pt"
        assert cfg.confidence_threshold == 0.35
        assert cfg.image_size == 640
        assert cfg.device is None
        assert "person" in cfg.obstacle_classes
        assert "car" in cfg.obstacle_classes

    def test_resolve_device_auto(self):
        from narrator.config import DetectorConfig
        cfg = DetectorConfig()
        device = cfg.resolve_device()
        # Should return "0" (CUDA) or "cpu" — either is valid.
        assert device in ("0", "cpu")

    def test_resolve_device_explicit(self):
        from narrator.config import DetectorConfig
        cfg = DetectorConfig(device="cpu")
        assert cfg.resolve_device() == "cpu"

    def test_obstacle_classes_are_independent(self):
        """Each config instance gets its own copy of obstacle classes."""
        from narrator.config import DetectorConfig
        a = DetectorConfig()
        b = DetectorConfig()
        a.obstacle_classes.add("zebra")
        assert "zebra" not in b.obstacle_classes


class TestSpeakerConfig:
    """Verify SpeakerConfig defaults."""

    def test_defaults(self):
        from narrator.config import SpeakerConfig
        cfg = SpeakerConfig()
        assert cfg.rate == 175
        assert cfg.cooldown_seconds == 2.0


class TestAppConfig:
    """Verify top-level AppConfig."""

    def test_defaults(self):
        from narrator.config import AppConfig
        cfg = AppConfig()
        assert cfg.detector.model_path == "yolo11n.pt"
        assert cfg.speaker.rate == 175

    def test_from_env(self, monkeypatch):
        from narrator.config import AppConfig
        monkeypatch.setenv("NARRATOR_CONFIDENCE", "0.50")
        monkeypatch.setenv("NARRATOR_DEVICE", "cpu")
        monkeypatch.setenv("NARRATOR_MODEL", "yolo11s.pt")
        cfg = AppConfig.from_env()
        assert cfg.detector.confidence_threshold == 0.50
        assert cfg.detector.device == "cpu"
        assert cfg.detector.model_path == "yolo11s.pt"

    def test_from_env_no_overrides(self, monkeypatch):
        """Without env vars, defaults are preserved."""
        from narrator.config import AppConfig
        monkeypatch.delenv("NARRATOR_CONFIDENCE", raising=False)
        monkeypatch.delenv("NARRATOR_DEVICE", raising=False)
        monkeypatch.delenv("NARRATOR_MODEL", raising=False)
        cfg = AppConfig.from_env()
        assert cfg.detector.confidence_threshold == 0.35
        assert cfg.detector.device is None


class TestProjectPaths:
    """Verify that PROJECT_ROOT points to the actual project directory."""

    def test_project_root_exists(self):
        from narrator.config import PROJECT_ROOT
        assert PROJECT_ROOT.is_dir()

    def test_project_root_contains_pyproject(self):
        from narrator.config import PROJECT_ROOT
        assert (PROJECT_ROOT / "pyproject.toml").is_file()
