"""Unit and integration tests for System Health Monitoring and Reliable Narration.

Covers:
- Camera failure, capture failure, and stale frame detection
- Detector exceptions vs valid zero detections
- False inference avoidance (zero detections != clear path, zero detections != lens blocked)
- Recursive audio failure prevention (no spoken warnings over dead audio)
- Health alert rate limiting and recovery announcements
- Non-interference: genuine obstacle alerts never suppressed by health alerts
- Performance measurement separation: capture latency, inference latency,
  submission latency, queue latency, and playback latency.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from narrator.config import AppConfig, DetectorConfig, HealthConfig, SpeakerConfig
from narrator.detector import Detection, Detector
from narrator.direction import Direction
from narrator.distance import Proximity
from narrator.events import EventNarrator, EventType, NarrationEvent
from narrator.health import ComponentHealthState, ComponentStatus, HealthMonitor
from narrator.metrics import PerformanceMonitor
from narrator.pipeline import Pipeline
from narrator.prioritizer import PrioritizedObstacle
from narrator.speaker import Speaker


# ==============================================================================
# 1. Camera Health & Stale Frames Tests
# ==============================================================================


class TestCameraHealthAndStaleFrames:
    def test_camera_open_failure_reports_unavailable_and_announces_warning(self) -> None:
        health = HealthMonitor(HealthConfig(warning_cooldown_seconds=30.0))
        phrase = health.report_camera_open_failed(current_time=10.0, error_msg="Device busy")

        assert health.get_status("camera") == ComponentStatus.UNAVAILABLE
        assert phrase == "Camera unavailable."
        assert not health.is_healthy()
        assert not health.is_perception_available()

    def test_camera_single_frame_failure_does_not_immediately_alert(self) -> None:
        """A single transient frame read drop should not trigger an immediate spoken alert."""
        health = HealthMonitor(HealthConfig(consecutive_capture_failures_to_alert=3))
        phrase = health.report_frame_capture_failure(current_time=1.0)

        # Still OK, waiting for confirmation
        assert health.get_status("camera") == ComponentStatus.OK
        assert phrase is None

    def test_camera_consecutive_frame_failures_confirm_failure(self) -> None:
        """Confirmed failure after N consecutive drops triggers spoken alert."""
        health = HealthMonitor(HealthConfig(consecutive_capture_failures_to_alert=3))

        health.report_frame_capture_failure(current_time=1.0)
        health.report_frame_capture_failure(current_time=1.03)
        phrase = health.report_frame_capture_failure(current_time=1.06)

        assert health.get_status("camera") == ComponentStatus.UNAVAILABLE
        assert phrase == "Camera unavailable."

    def test_camera_recovery_after_failure(self) -> None:
        """Successful frame read after failure announces recovery."""
        health = HealthMonitor(HealthConfig(consecutive_capture_failures_to_alert=2))

        health.report_frame_capture_failure(current_time=1.0)
        health.report_frame_capture_failure(current_time=1.03)
        assert health.get_status("camera") == ComponentStatus.UNAVAILABLE

        # Frame read succeeds
        recovery_phrase = health.report_frame_capture_success(current_time=2.0)
        assert health.get_status("camera") == ComponentStatus.OK
        assert recovery_phrase == "Camera recovered."

    def test_stale_frame_detection_when_interval_exceeds_threshold(self) -> None:
        """Frames not received within stale_frame_threshold_seconds trigger stale state."""
        health = HealthMonitor(HealthConfig(stale_frame_threshold_seconds=2.0))

        health.report_frame_capture_success(current_time=1.0)
        assert health.get_status("camera") == ComponentStatus.OK

        # 2.5 seconds later with no frames, check stale
        phrase = health.check_stale_frame(current_time=3.5)
        assert health.get_status("camera") == ComponentStatus.STALE
        assert phrase == "Camera feed stale."

    def test_stale_frame_recovers_on_next_valid_frame(self) -> None:
        health = HealthMonitor(HealthConfig(stale_frame_threshold_seconds=2.0))

        health.report_frame_capture_success(current_time=1.0)
        health.check_stale_frame(current_time=3.5)
        assert health.get_status("camera") == ComponentStatus.STALE

        # Fresh frame arrives
        phrase = health.report_frame_capture_success(current_time=3.6)
        assert health.get_status("camera") == ComponentStatus.OK
        assert phrase == "Camera recovered."


# ==============================================================================
# 2. Detector Health vs. Valid Empty Detections
# ==============================================================================


class TestDetectorHealthVsEmptyDetections:
    def test_detector_exception_triggers_unavailable_alert(self) -> None:
        health = HealthMonitor()
        phrase = health.report_detector_error(RuntimeError("CUDA out of memory"), current_time=5.0)

        assert health.get_status("detector") == ComponentStatus.UNAVAILABLE
        assert phrase == "Obstacle detection unavailable."
        assert not health.is_healthy()
        assert not health.is_perception_available()

    def test_detector_recovery_after_exception(self) -> None:
        health = HealthMonitor()
        health.report_detector_error(RuntimeError("CUDA out of memory"), current_time=5.0)
        assert health.get_status("detector") == ComponentStatus.UNAVAILABLE

        # Next inference succeeds
        phrase = health.report_detector_success(detection_count=2, current_time=6.0)
        assert health.get_status("detector") == ComponentStatus.OK
        assert phrase == "Obstacle detection recovered."
        assert health.is_healthy()

    def test_zero_detections_on_valid_frame_is_not_an_error(self) -> None:
        """Zero detections on a valid frame is normal operation, NOT a failure."""
        health = HealthMonitor()

        phrase = health.report_detector_success(detection_count=0, current_time=10.0)

        assert health.get_status("detector") == ComponentStatus.OK
        assert phrase is None
        assert health.consecutive_zero_detections == 1
        assert health.is_healthy()

    def test_low_detection_count_does_not_imply_failure(self) -> None:
        """A single detection (low count) is a completely healthy inference."""
        health = HealthMonitor()

        phrase = health.report_detector_success(detection_count=1, current_time=1.0)

        assert health.get_status("detector") == ComponentStatus.OK
        assert phrase is None
        assert health.consecutive_zero_detections == 0
        assert health.is_healthy()

    def test_detector_class_preserves_last_error_and_is_healthy_flag(self) -> None:
        cfg = DetectorConfig()
        with patch.object(Detector, "_load_model"):
            detector = Detector(cfg)
            assert detector.is_healthy is True
            assert detector.last_error is None

            # Simulate exception during detect
            detector._model = MagicMock()
            detector._model.track.side_effect = RuntimeError("Inference crash")

            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            dets = detector.detect(frame)

            assert dets == []
            assert detector.is_healthy is False
            assert isinstance(detector.last_error, RuntimeError)
            assert detector.last_detection_count == 0


# ==============================================================================
# 3. Speaker Health & Recursive Alert Avoidance
# ==============================================================================


class TestSpeakerHealthAndRecursiveAlertSuppression:
    def test_speaker_worker_failure_tracked_without_spoken_alert(self) -> None:
        """Dead speaker status is recorded, but never emits a spoken phrase to avoid dead-audio loops."""
        health = HealthMonitor()

        health.report_speaker_status(is_worker_alive=False, queue_size=0, current_time=1.0)

        assert health.get_status("speaker") == ComponentStatus.UNAVAILABLE
        assert not health.is_healthy()
        # Ensure phrases for speaker transitions are completely empty
        assert health.WARNING_PHRASES.get("speaker") == {}

    def test_speaker_worker_recovery_tracked(self) -> None:
        health = HealthMonitor()
        health.report_speaker_status(is_worker_alive=False, queue_size=0, current_time=1.0)
        assert health.get_status("speaker") == ComponentStatus.UNAVAILABLE

        health.report_speaker_status(is_worker_alive=True, queue_size=0, current_time=2.0)
        assert health.get_status("speaker") == ComponentStatus.OK
        assert health.is_healthy()

    def test_speaker_submission_rejection_tracking(self) -> None:
        health = HealthMonitor()
        health.report_speech_submission(accepted=False, reason="cooldown")
        snapshot = health.get_snapshot()
        assert snapshot.speaker_status == ComponentStatus.OK  # Rejection alone is not a worker failure


# ==============================================================================
# 4. Health Warning Rate Limiting Tests
# ==============================================================================


class TestHealthWarningRateLimiting:
    def test_repeated_failure_transitions_suppressed_by_cooldown(self) -> None:
        health = HealthMonitor(HealthConfig(warning_cooldown_seconds=30.0))

        # First alert at t=10.0
        phrase1 = health.report_camera_open_failed(current_time=10.0)
        assert phrase1 == "Camera unavailable."

        # Recover at t=15.0
        phrase2 = health.report_frame_capture_success(current_time=15.0)
        assert phrase2 == "Camera recovered."

        # Fail again at t=20.0 (< 30s since phrase1) - state changes to UNAVAILABLE but speech is suppressed
        phrase3 = health.report_camera_open_failed(current_time=20.0)
        assert phrase3 is None  # Suppressed by rate limit!

        # Recover again at t=25.0
        phrase3_rec = health.report_frame_capture_success(current_time=25.0)
        assert phrase3_rec == "Camera recovered."

        # Fail again after cooldown at t=45.0 (> 30s since phrase1)
        phrase4 = health.report_camera_open_failed(current_time=45.0)
        assert phrase4 == "Camera unavailable."


# ==============================================================================
# 5. Non-Interference: Obstacle Warnings Never Blocked by Health Alerts
# ==============================================================================


class TestObstacleWarningsNeverSuppressedByHealth:
    def test_genuine_obstacle_alert_takes_priority_over_recovery_alert(self) -> None:
        """If an obstacle event occurs in the same frame as a detector recovery, obstacle alert speaks."""
        mock_detector = MagicMock()
        mock_detector.is_healthy = True
        mock_detector.last_error = None
        mock_detector.detect.return_value = [
            Detection("chair", 0.9, 200, 200, 440, 460, track_id=1)
        ]

        mock_speaker = MagicMock()
        mock_speaker.is_worker_alive = True
        mock_speaker.queue_size = 0
        mock_speaker.speak.return_value = True

        config = AppConfig()
        config.events.min_frames_to_confirm = 1
        pipeline = Pipeline(config, detector=mock_detector, speaker=mock_speaker)

        # Force detector into UNAVAILABLE state initially
        pipeline.health.report_detector_error("Previous error", current_time=1.0)
        assert pipeline.health.get_status("detector") == ComponentStatus.UNAVAILABLE

        # Process frame with obstacle: detector recovers AND detects obstacle
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with patch("time.monotonic", return_value=5.0):
            pipeline.process_frame(frame)

        # The obstacle alert must be spoken!
        mock_speaker.speak.assert_called_once_with("near chair ahead")
        # And detector status must now be recovered
        assert pipeline.health.get_status("detector") == ComponentStatus.OK

    def test_detector_failure_speaks_health_alert_when_no_obstacle_events(self) -> None:
        """When detector fails and no detections exist, health alert is announced."""
        mock_detector = MagicMock()
        mock_detector.is_healthy = False
        mock_detector.last_error = RuntimeError("Inference timeout")
        mock_detector.detect.return_value = []

        mock_speaker = MagicMock()
        mock_speaker.is_worker_alive = True
        mock_speaker.queue_size = 0
        mock_speaker.speak.return_value = True

        config = AppConfig()
        pipeline = Pipeline(config, detector=mock_detector, speaker=mock_speaker)

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with patch("time.monotonic", return_value=10.0):
            pipeline.process_frame(frame)

        # Urgent health alert spoken
        mock_speaker.speak.assert_called_once_with(
            "Obstacle detection unavailable.", urgent=True
        )


# ==============================================================================
# 6. Performance Measurements Separation Tests
# ==============================================================================


class TestNarrationPerformanceMetrics:
    def test_performance_monitor_records_and_separates_latencies(self) -> None:
        monitor = PerformanceMonitor(window_size=10, warmup_frames=2)

        # Frame 1 (warmup: index 0 discarded)
        monitor.record_frame(
            inference_latency_ms=10.0,
            total_latency_ms=20.0,
            active_tracks=2,
            capture_latency_ms=8.0,
            submission_latency_ms=4.0,
        )
        # Frame 2 (retained: index 1)
        monitor.record_frame(
            inference_latency_ms=16.0,
            total_latency_ms=26.0,
            active_tracks=2,
            capture_latency_ms=7.0,
            submission_latency_ms=5.0,
        )
        # Frame 3 (retained: index 2)
        monitor.record_frame(
            inference_latency_ms=14.0,
            total_latency_ms=22.0,
            active_tracks=1,
            capture_latency_ms=5.0,
            submission_latency_ms=3.0,
        )

        summary = monitor.get_summary()

        # Average of retained frames: (16 + 14) / 2 = 15.0
        assert summary["avg_inference_ms"] == 15.0
        # Average of retained frames: (26 + 22) / 2 = 24.0
        assert summary["avg_total_ms"] == 24.0
        # Average of retained frames: (7 + 5) / 2 = 6.0
        assert summary["avg_capture_ms"] == 6.0
        # Submission latency averaged across all submitted events: (4 + 5 + 3) / 3 = 4.0
        assert summary["avg_submission_ms"] == 4.0
        assert summary["active_tracks"] == 1

    def test_speaker_exposes_queue_latency_and_playback_duration(self) -> None:
        class FakeEngine:
            def __init__(self) -> None:
                pass

            def setProperty(self, name: str, value: object) -> None:
                pass

            def say(self, text: str) -> None:
                pass

            def runAndWait(self) -> None:
                time.sleep(0.01)

            def stop(self) -> None:
                pass

        cfg = SpeakerConfig(enabled=True, cooldown_seconds=0.0)
        with patch("pyttsx3.init", return_value=FakeEngine()):
            spk = Speaker(cfg)
            try:
                assert spk.speak("test alert") is True
                # Wait for worker thread to process (up to 1.0s)
                deadline = time.time() + 1.0
                while spk.last_playback_duration_ms == 0.0 and time.time() < deadline:
                    time.sleep(0.02)

                assert spk.last_queue_latency_ms >= 0.0
                assert spk.last_playback_duration_ms > 0.0
                stats = spk.submissions_stats
                assert stats["accepted"] == 1
                assert stats["total"] == 1
                assert stats["rejected_cooldown"] == 0
                assert stats["rejected_queue_full"] == 0
            finally:
                spk.stop()

    def test_narration_event_records_creation_time_for_submission_latency(self) -> None:
        t_create = 100.5
        with patch("time.monotonic", return_value=t_create):
            event = NarrationEvent(
                track_id=1,
                label="car",
                direction=Direction.CENTER,
                proximity=Proximity.NEAR,
                event_type=EventType.NEW_OBJECT,
                urgency_score=80.0,
                speech_text="near car ahead",
            )
            assert event.created_at == t_create


# ==============================================================================
# 7. Pipeline Health Integration Tests
# ==============================================================================


class TestPipelineHealthIntegration:
    def test_pipeline_run_announces_warning_and_raises_when_camera_fails(self) -> None:
        config = AppConfig()
        mock_speaker = MagicMock()
        mock_speaker.is_worker_alive = True
        mock_speaker.queue_size = 0
        pipeline = Pipeline(config, speaker=mock_speaker)

        with patch("cv2.VideoCapture") as mock_cap_cls:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = False
            mock_cap_cls.return_value = mock_cap

            with patch("time.sleep"):  # prevent sleep during test
                with pytest.raises(RuntimeError, match="Failed to open camera"):
                    pipeline.run()

            # Verify camera unavailable spoken alert was attempted
            mock_speaker.speak.assert_called_once_with("Camera unavailable.", urgent=True)
            assert pipeline.health.get_status("camera") == ComponentStatus.UNAVAILABLE

    def test_pipeline_process_frame_handles_empty_or_none_frame(self) -> None:
        mock_speaker = MagicMock()
        pipeline = Pipeline(AppConfig(), speaker=mock_speaker)

        empty_frame = np.array([])
        annotated, obstacles, latency_ms = pipeline.process_frame(empty_frame)

        assert obstacles == []
        assert latency_ms == 0.0
        assert pipeline.health.get_status("camera") == ComponentStatus.OK  # 1 failure < threshold
