"""Unit tests for Stage 2 core modules.

Covers:
- Detection dataclass and bounding box properties
- Direction estimation and boundary classification
- Proximity estimation and frame-height thresholds
- Hazard prioritization, sorting, and speech string formatting
- Speaker cooldown and non-blocking queue behavior (with mocked engine)
- Pipeline single-frame processing and annotations (with mocked detector/speaker)
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from narrator.config import AppConfig, DetectorConfig, SpeakerConfig
from narrator.detector import Detection
from narrator.direction import Direction, estimate_direction
from narrator.distance import Proximity, estimate_proximity
from narrator.pipeline import Pipeline
from narrator.prioritizer import (
    PrioritizedObstacle,
    prioritize_detections,
)
from narrator.speaker import Speaker


# ==============================================================================
# Detection Tests
# ==============================================================================


class TestDetection:
    def test_bounding_box_geometry(self) -> None:
        det = Detection(
            label="car",
            confidence=0.92,
            x1=100.0,
            y1=50.0,
            x2=300.0,
            y2=250.0,
        )
        assert det.cx == 200.0
        assert det.cy == 150.0
        assert det.width == 200.0
        assert det.height == 200.0

    def test_detection_immutable(self) -> None:
        det = Detection("person", 0.8, 10, 20, 50, 100)
        with pytest.raises(Exception):
            det.label = "dog"  # type: ignore[misc]


# ==============================================================================
# Direction Tests
# ==============================================================================


class TestDirection:
    @pytest.mark.parametrize(
        ("cx", "frame_width", "expected"),
        [
            (100.0, 640, Direction.LEFT),      # 100/640 = 0.156 < 0.33
            (211.0, 640, Direction.LEFT),      # 211/640 = 0.329 < 0.33
            (320.0, 640, Direction.CENTER),    # 320/640 = 0.50 (ahead)
            (212.0, 640, Direction.CENTER),    # boundary
            (428.0, 640, Direction.CENTER),    # 428/640 = 0.668 <= 0.67
            (500.0, 640, Direction.RIGHT),     # 500/640 = 0.781 > 0.67
            (639.0, 640, Direction.RIGHT),
        ],
    )
    def test_estimate_direction_defaults(
        self, cx: float, frame_width: int, expected: Direction
    ) -> None:
        assert estimate_direction(cx, frame_width) == expected

    def test_estimate_direction_zero_width(self) -> None:
        assert estimate_direction(100, 0) == Direction.CENTER

    def test_custom_boundaries(self) -> None:
        # Custom: left < 0.2, right > 0.8
        assert estimate_direction(150, 1000, left_boundary=0.2, right_boundary=0.8) == Direction.LEFT
        assert estimate_direction(500, 1000, left_boundary=0.2, right_boundary=0.8) == Direction.CENTER
        assert estimate_direction(850, 1000, left_boundary=0.2, right_boundary=0.8) == Direction.RIGHT


# ==============================================================================
# Distance / Proximity Tests
# ==============================================================================


class TestProximity:
    @pytest.mark.parametrize(
        ("box_height", "frame_height", "expected"),
        [
            (300.0, 480, Proximity.NEAR),      # 300/480 = 0.625 >= 0.50
            (240.0, 480, Proximity.NEAR),      # 240/480 = 0.500 >= 0.50
            (150.0, 480, Proximity.MEDIUM),    # 150/480 = 0.312 >= 0.20
            (96.0, 480, Proximity.MEDIUM),     # 96/480 = 0.200 >= 0.20
            (50.0, 480, Proximity.FAR),        # 50/480 = 0.104 < 0.20
            (10.0, 480, Proximity.FAR),
        ],
    )
    def test_estimate_proximity_defaults(
        self, box_height: float, frame_height: int, expected: Proximity
    ) -> None:
        assert estimate_proximity(box_height, frame_height) == expected

    def test_estimate_proximity_zero_height(self) -> None:
        assert estimate_proximity(100, 0) == Proximity.FAR

    def test_custom_thresholds(self) -> None:
        assert (
            estimate_proximity(400, 1000, near_threshold=0.6, medium_threshold=0.3)
            == Proximity.MEDIUM
        )
        assert (
            estimate_proximity(700, 1000, near_threshold=0.6, medium_threshold=0.3)
            == Proximity.NEAR
        )


# ==============================================================================
# Prioritizer Tests
# ==============================================================================


class TestPrioritizer:
    def test_empty_detections(self) -> None:
        assert prioritize_detections([], frame_width=640, frame_height=480) == []

    def test_proximity_ranking_dominates(self) -> None:
        # A FAR car directly ahead vs a NEAR person on the side
        # NEAR weight = 100, FAR weight = 10
        far_car = Detection("car", 0.9, 300, 200, 340, 250)  # height=50/480 (FAR), center
        near_person = Detection("person", 0.8, 20, 10, 100, 350)  # height=340/480 (NEAR), left

        ranked = prioritize_detections([far_car, near_person], frame_width=640, frame_height=480)
        assert len(ranked) == 2
        # near person should rank higher than far car despite car category weight
        assert ranked[0].label == "person"
        assert ranked[0].proximity == Proximity.NEAR
        assert ranked[1].label == "car"
        assert ranked[1].proximity == Proximity.FAR

    def test_center_ranked_above_peripheral_same_proximity(self) -> None:
        # Two MEDIUM obstacles: one center, one right
        center_bike = Detection("bicycle", 0.85, 280, 100, 360, 250)  # height 150/480, center
        right_bike = Detection("bicycle", 0.85, 500, 100, 580, 250)   # height 150/480, right

        ranked = prioritize_detections([right_bike, center_bike], frame_width=640, frame_height=480)
        assert ranked[0].direction == Direction.CENTER
        assert ranked[1].direction == Direction.RIGHT

    def test_speech_text_formatting(self) -> None:
        det_near = Detection("car", 0.9, 250, 10, 390, 350)  # NEAR, ahead
        det_far = Detection("chair", 0.7, 50, 200, 80, 250)  # FAR, left

        ranked = prioritize_detections([det_near, det_far], frame_width=640, frame_height=480)

        # "near car ahead"
        assert ranked[0].speech_text == "near car ahead"
        # "chair left" (no "far" prefix per concise design)
        assert ranked[1].speech_text == "chair left"


# ==============================================================================
# Speaker Cooldown and Queue Tests
# ==============================================================================


class TestSpeaker:
    def test_speaker_disabled(self) -> None:
        cfg = SpeakerConfig(enabled=False)
        spk = Speaker(cfg)
        assert not spk.speak("car ahead")
        assert not spk.is_on_cooldown("car ahead")
        spk.stop()

    def test_empty_string_ignored(self) -> None:
        cfg = SpeakerConfig(enabled=True)
        # Mock worker to prevent starting actual TTS engine
        with patch.object(Speaker, "_start_worker"):
            spk = Speaker(cfg)
            assert not spk.speak("   ")
            spk.stop()

    def test_cooldown_suppresses_duplicates(self) -> None:
        cfg = SpeakerConfig(enabled=True, cooldown_seconds=2.0)
        with patch.object(Speaker, "_start_worker"):
            spk = Speaker(cfg)

            # First message should be accepted
            assert spk.speak("near car ahead") is True
            assert spk.is_on_cooldown("near car ahead") is True

            # Immediate repeat must be suppressed
            assert spk.speak("near car ahead") is False

            # Different message should still be accepted
            assert spk.speak("bicycle left") is True

            # After cooldown expires, original message accepted again
            with patch("time.monotonic", return_value=time.monotonic() + 3.0):
                assert spk.is_on_cooldown("near car ahead") is False
                assert spk.speak("near car ahead") is True

            spk.stop()

    def test_queue_full_drops_gracefully(self) -> None:
        cfg = SpeakerConfig(enabled=True, cooldown_seconds=0.0)  # No cooldown
        with patch.object(Speaker, "_start_worker"):
            spk = Speaker(cfg)
            # Fill the 4-slot queue
            for i in range(4):
                assert spk.speak(f"msg {i}") is True
            # 5th item should be dropped cleanly without blocking
            assert spk.speak("overflow message") is False
            spk.stop()

    def test_worker_loop_processes_multiple_utterances_with_fake_engine(self) -> None:
        """Worker-level test using a fake TTS engine verifying continuous queue processing."""
        spoken_utterances: list[str] = []

        class FakeEngine:
            def __init__(self) -> None:
                pass

            def setProperty(self, name: str, value: object) -> None:
                pass

            def say(self, text: str) -> None:
                spoken_utterances.append(text)

            def runAndWait(self) -> None:
                pass

            def stop(self) -> None:
                pass

        fake_engine = FakeEngine()
        cfg = SpeakerConfig(enabled=True, cooldown_seconds=0.0)

        with patch("pyttsx3.init", return_value=fake_engine):
            spk = Speaker(cfg)
            try:
                assert spk.is_worker_alive is True

                # Send 3 distinct utterances
                assert spk.speak("first obstacle ahead") is True
                assert spk.speak("second obstacle left") is True
                assert spk.speak("third obstacle right") is True

                # Wait for worker thread to drain queue
                timeout = 2.0
                start_t = time.monotonic()
                while len(spoken_utterances) < 3 and (time.monotonic() - start_t) < timeout:
                    time.sleep(0.02)

                assert spoken_utterances == [
                    "first obstacle ahead",
                    "second obstacle left",
                    "third obstacle right",
                ]
                assert spk.queue_size == 0
                assert spk.is_worker_alive is True
            finally:
                spk.stop()

        assert spk.is_worker_alive is False


# ==============================================================================
# Pipeline Single Frame Tests
# ==============================================================================


class TestPipeline:
    def test_process_frame_synthetic(self) -> None:
        # Create a mock detector returning a known detection
        mock_detector = MagicMock()
        mock_detector.detect.return_value = [
            Detection("car", 0.95, 200, 50, 440, 300)
        ]

        # Create a mock speaker
        mock_speaker = MagicMock()

        # Use min_frames_to_confirm=1 so NEW_OBJECT fires on first frame
        # Disable depth to test pure heuristic proximity behavior
        config = AppConfig()
        config.events.min_frames_to_confirm = 1
        config.depth.enabled = False
        pipeline = Pipeline(config, detector=mock_detector, speaker=mock_speaker)

        # 480x640 synthetic black image
        synthetic_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        annotated, obstacles, latency_ms = pipeline.process_frame(synthetic_frame)

        assert len(obstacles) == 1
        assert obstacles[0].label == "car"
        assert latency_ms >= 0.0
        assert annotated.shape == (480, 640, 3)

        # Assert speaker received NEW_OBJECT event speech text
        # With min_frames_to_confirm=1, first frame triggers NEW_OBJECT
        # Car at x=200-440 (CENTER), height=250/480=0.52 (NEAR) -> "near car ahead"
        mock_speaker.speak.assert_called_once_with("near car ahead")

    def test_process_frame_empty_detections(self) -> None:
        mock_detector = MagicMock()
        mock_detector.detect.return_value = []
        mock_speaker = MagicMock()

        pipeline = Pipeline(AppConfig(), detector=mock_detector, speaker=mock_speaker)
        synthetic_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        annotated, obstacles, _ = pipeline.process_frame(synthetic_frame)

        assert len(obstacles) == 0
        mock_speaker.speak.assert_not_called()
        assert annotated.shape == (480, 640, 3)
