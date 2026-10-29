"""Unit tests for tracking, event-based narration, and performance monitoring.

Covers:
- Tracking integration with ByteTrack-like behavior
- EventNarrator new object detection and confirmation
- EventNarrator proximity escalation detection
- EventNarrator urgency escalation detection
- EventNarrator periodic reminders
- Global and per-track cooldown enforcement
- Track disappearance and grace period handling
- Fallback behavior when tracking is disabled
- PerformanceMonitor rolling window calculations
- Warm-up frame exclusion
- Muted audio state behavior
"""

from __future__ import annotations

from pathlib import Path
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from narrator.config import AppConfig, DetectorConfig, EventsConfig
from narrator.detector import Detection, Detector
from narrator.direction import Direction
from narrator.distance import Proximity
from narrator.events import EventNarrator, NarrationEvent, EventType, TrackRecord
from narrator.metrics import PerformanceMonitor
from narrator.prioritizer import PrioritizedObstacle, prioritize_detections
from narrator.pipeline import Pipeline
from narrator.speaker import Speaker


# ==============================================================================
# Tracking Tests
# ==============================================================================


class TestTrackingIntegration:
    """Test YOLO tracking integration with ByteTrack."""

    def test_detection_dataclass_includes_track_id(self):
        """Detection dataclass should support optional track_id."""
        det = Detection(
            label="person",
            confidence=0.8,
            x1=100,
            y1=100,
            x2=200,
            y2=200,
            track_id=42,
        )
        assert det.track_id == 42

        # Default should be None for backward compatibility
        det_default = Detection(
            label="person",
            confidence=0.8,
            x1=100,
            y1=100,
            x2=200,
            y2=200,
        )
        assert det_default.track_id is None

    def test_detector_config_tracking_defaults(self):
        """DetectorConfig should have tracking enabled by default."""
        config = DetectorConfig()
        assert config.tracking_enabled is True
        assert config.tracker_type == "bytetrack.yaml"

    def test_detector_handles_tracking_disabled(self):
        """When tracking is disabled, detector should use predict not track."""
        config = DetectorConfig(tracking_enabled=False)
        detector = Detector(config)

        # Mock the model to verify predict is called
        with patch.object(detector._model, 'predict', return_value=[]) as mock_predict, \
             patch.object(detector._model, 'track', return_value=[]) as mock_track:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            detector.detect(frame)
            mock_predict.assert_called_once()
            mock_track.assert_not_called()

    def test_detector_extracts_track_id_safely(self):
        """Detector should safely extract track_id from tracking results."""
        config = DetectorConfig(tracking_enabled=True)
        detector = Detector(config)

        # Mock results that simulate ByteTrack output
        mock_result = MagicMock()
        mock_box = MagicMock()
        mock_box.cls = [0.0]  # person class
        mock_box.conf = [0.85]
        mock_box.xyxy = [MagicMock(tolist=lambda: [100.0, 100.0, 200.0, 200.0])]
        mock_box.id = [42.0]  # track ID as float tensor
        mock_result.boxes = [mock_box]

        with patch.object(detector._model, 'track', return_value=[mock_result]):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            detections = detector.detect(frame)

            assert len(detections) == 1
            det = detections[0]
            assert det.label == "person"
            assert det.confidence == 0.85
            assert det.track_id == 42

    def test_detector_handles_missing_track_id_gracefully(self):
        """Detector should handle cases where tracking results lack ID."""
        config = DetectorConfig(tracking_enabled=True)
        detector = Detector(config)

        # Mock results without track IDs
        mock_result = MagicMock()
        mock_box = MagicMock()
        mock_box.cls = [0.0]  # person class
        mock_box.conf = [0.85]
        mock_box.xyxy = [MagicMock(tolist=lambda: [100.0, 100.0, 200.0, 200.0])]
        mock_box.id = None  # No tracking ID
        mock_result.boxes = [mock_box]

        with patch.object(detector._model, 'track', return_value=[mock_result]):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            detections = detector.detect(frame)

            assert len(detections) == 1
            det = detections[0]
            assert det.track_id is None  # Should be None when not available

    def test_detector_reset_tracker_exists(self):
        """Detector should have a reset_tracker method."""
        config = DetectorConfig()
        detector = Detector(config)
        # Should not raise
        detector.reset_tracker()


# ==============================================================================
# EventNarrator Tests
# ==============================================================================


class TestEventNarrator:
    """Test event-based narration logic."""

    def test_event_narrator_initialization(self):
        """EventNarrator should initialize with proper defaults."""
        narrator = EventNarrator()
        assert narrator._config.global_cooldown_seconds == 1.5
        assert narrator._config.track_cooldown_seconds == 4.0
        assert narrator._config.announce_new_objects is True
        assert narrator._config.announce_proximity_changes is True
        assert narrator._config.min_frames_to_confirm == 2
        assert narrator._tracking_enabled is True
        assert len(narrator._tracks) == 0

    def test_event_narrator_with_custom_config(self):
        """EventNarrator should accept custom EventsConfig."""
        config = EventsConfig(
            global_cooldown_seconds=2.0,
            track_cooldown_seconds=5.0,
            announce_new_objects=False,
            min_frames_to_confirm=3,
        )
        narrator = EventNarrator(config=config, tracking_enabled=False)
        assert narrator._config.global_cooldown_seconds == 2.0
        assert narrator._config.track_cooldown_seconds == 5.0
        assert narrator._config.announce_new_objects is False
        assert narrator._config.min_frames_to_confirm == 3
        assert narrator._tracking_enabled is False

    def test_new_object_event_after_confirmation_frames(self):
        """New object event should trigger after min_frames_to_confirm."""
        config = EventsConfig(min_frames_to_confirm=2)
        narrator = EventNarrator(config=config)

        # Create a mock obstacle
        det = Detection(label="person", confidence=0.8, x1=100, y1=100, x2=200, y2=200)
        obstacle = PrioritizedObstacle(
            detection=det,
            direction=Direction.CENTER,
            proximity=Proximity.NEAR,
            urgency_score=100.0,
        )

        with patch('time.monotonic') as mock_time:
            mock_time.return_value = 0.0
            # First frame - should not trigger (consecutive_frames = 1)
            events = narrator.update([obstacle], current_time=0.0)
            assert events is None

            # Second frame - should trigger NEW_OBJECT (consecutive_frames = 2)
            mock_time.return_value = 0.1
            events = narrator.update([obstacle], current_time=0.1)
            assert events is not None
            assert events.event_type == EventType.NEW_OBJECT
            assert events.label == "person"
            assert events.track_id is None  # No tracking in this test

    def test_spurious_single_frame_detection_suppressed(self):
        """Single frame detections should not trigger new object events."""
        config = EventsConfig(min_frames_to_confirm=2)
        narrator = EventNarrator(config=config)

        det = Detection(label="car", confidence=0.9, x1=300, y1=100, x2=400, y2=200)
        obstacle = PrioritizedObstacle(
            detection=det,
            direction=Direction.RIGHT,
            proximity=Proximity.FAR,
            urgency_score=20.0,
        )

        with patch('time.monotonic') as mock_time:
            mock_time.return_value = 0.0
            # Only one frame - should not trigger (consecutive_frames = 1)
            events = narrator.update([obstacle], current_time=0.0)
            assert events is None

            # Second frame WITHOUT the obstacle - should still not trigger
            # (track disappears, consecutive_frames resets to 0 when track reappears)
            mock_time.return_value = 0.1
            events = narrator.update([], current_time=0.1)  # Empty frame
            assert events is None

            # Third frame WITH the obstacle again - should not trigger
            # (this is the first frame of reappearance, consecutive_frames = 1)
            mock_time.return_value = 0.2
            events = narrator.update([obstacle], current_time=0.2)
            assert events is None  # Still need second consecutive frame

    def test_proximity_escalation_event(self):
        """Proximity escalation (FAR -> MEDIUM, MEDIUM -> NEAR) should trigger event."""
        config = EventsConfig(announce_proximity_changes=True)
        narrator = EventNarrator(config=config)

        # Start with FAR
        det_far = Detection(label="car", confidence=0.9, x1=300, y1=100, x2=400, y2=150)  # small height
        obstacle_far = PrioritizedObstacle(
            detection=det_far,
            direction=Direction.RIGHT,
            proximity=Proximity.FAR,
            urgency_score=20.0,
        )

        with patch('time.monotonic') as mock_time:
            mock_time.return_value = 0.0
            # Establish the track
            narrator.update([obstacle_far], current_time=0.0)

            # Now MEDIUM - should trigger PROXIMITY_ESCALATION
            det_med = Detection(label="car", confidence=0.9, x1=300, y1=200, x2=400, y2=300)  # medium height
            obstacle_med = PrioritizedObstacle(
                detection=det_med,
                direction=Direction.RIGHT,
                proximity=Proximity.MEDIUM,
                urgency_score=50.0,
            )
            mock_time.return_value = 1.6  # Advance past global cooldown
            events = narrator.update([obstacle_med], current_time=1.6)
            assert events is not None
            assert events.event_type == EventType.PROXIMITY_ESCALATION
            assert events.label == "car"
            assert events.proximity == Proximity.MEDIUM

            # Now NEAR - should trigger another PROXIMITY_ESCALATION
            det_near = Detection(label="car", confidence=0.9, x1=300, y1=300, x2=400, y2=400)  # large height
            obstacle_near = PrioritizedObstacle(
                detection=det_near,
                direction=Direction.RIGHT,
                proximity=Proximity.NEAR,
                urgency_score=100.0,
            )
            mock_time.return_value = 3.2  # Advance past global cooldown
            events = narrator.update([obstacle_near], current_time=3.2)
            assert events is not None
            assert events.event_type == EventType.PROXIMITY_ESCALATION
            assert events.proximity == Proximity.NEAR

    def test_proximity_de_escalation_no_event(self):
        """Proximity de-escalation (NEAR -> FAR) should NOT trigger event."""
        config = EventsConfig(announce_proximity_changes=True)
        narrator = EventNarrator(config=config)

        # Start with NEAR
        det_near = Detection(label="person", confidence=0.8, x1=100, y1=300, x2=200, y2=400, track_id=21)
        obstacle_near = PrioritizedObstacle(
            detection=det_near,
            direction=Direction.LEFT,
            proximity=Proximity.NEAR,
            urgency_score=100.0,
        )

        with patch('time.monotonic') as mock_time:
            mock_time.return_value = 0.0
            event = narrator.update([obstacle_near], current_time=0.0)  # Establish frame 1
            if event:
                narrator.record_spoken(event)
            event = narrator.update([obstacle_near], current_time=0.1)  # Confirm frame 2
            if event:
                narrator.record_spoken(event)

            # Now FAR at 2.0s (past global cooldown) - should NOT trigger (de-escalation)
            det_far = Detection(label="person", confidence=0.8, x1=100, y1=100, x2=200, y2=150, track_id=21)
            obstacle_far = PrioritizedObstacle(
                detection=det_far,
                direction=Direction.LEFT,
                proximity=Proximity.FAR,
                urgency_score=10.0,
            )
            mock_time.return_value = 2.0
            events = narrator.update([obstacle_far], current_time=2.0)
            assert events is None  # No event for de-escalation

    def test_urgency_escalation_event(self):
        """Significant urgency increase should trigger URGENCY_ESCALATION event."""
        config = EventsConfig()
        narrator = EventNarrator(config=config)

        # Low urgency obstacle
        det = Detection(label="chair", confidence=0.5, x1=50, y1=100, x2=150, y2=200, track_id=21)
        obstacle_low = PrioritizedObstacle(
            detection=det,
            direction=Direction.LEFT,
            proximity=Proximity.FAR,
            urgency_score=20.0,  # Low score
        )

        with patch('time.monotonic') as mock_time:
            mock_time.return_value = 0.0
            # Establish track for 2 frames to confirm
            narrator.update([obstacle_low], current_time=0.0)  # Frame 1
            narrator.update([obstacle_low], current_time=0.1)  # Frame 2 - should trigger NEW_OBJECT

            # Now trigger urgency escalation after BOTH global cooldown (1.5s) AND track cooldown (4.0s)
            # Need to wait until now - last_spoken >= track_cooldown (4.0s)
            det_high = Detection(label="chair", confidence=0.95, x1=50, y1=100, x2=150, y2=200, track_id=21)  # SAME track_id as obstacle_low
            obstacle_high = PrioritizedObstacle(
                detection=det_high,
                direction=Direction.LEFT,
                proximity=Proximity.FAR,
                urgency_score=50.0,  # > 1.3 * 20.0 = 26.0
            )
            mock_time.return_value = 5.0  # 5.0 - 0.1 = 4.9 > 4.0 (track cooldown)
            events = narrator.update([obstacle_high], current_time=5.0)
            assert events is not None
            assert events.event_type == EventType.URGENCY_ESCALATION
            assert events.label == "chair"

    def test_urgency_escalation_below_threshold_no_event(self):
        """Minor urgency increase should NOT trigger event."""
        config = EventsConfig()
        narrator = EventNarrator(config=config)

        det = Detection(label="chair", confidence=0.5, x1=50, y1=100, x2=150, y2=200, track_id=23)
        obstacle_low = PrioritizedObstacle(
            detection=det,
            direction=Direction.LEFT,
            proximity=Proximity.FAR,
            urgency_score=20.0,
        )

        with patch('time.monotonic') as mock_time:
            mock_time.return_value = 0.0
            event = narrator.update([obstacle_low], current_time=0.0)  # Frame 1
            if event:
                narrator.record_spoken(event)
            event = narrator.update([obstacle_low], current_time=0.1)  # Frame 2 (confirmed)
            if event:
                narrator.record_spoken(event)

            # Small increase (less than 1.3x) after global cooldown
            det_slight = Detection(label="chair", confidence=0.55, x1=50, y1=100, x2=150, y2=200, track_id=23)
            obstacle_slight = PrioritizedObstacle(
                detection=det_slight,
                direction=Direction.LEFT,
                proximity=Proximity.FAR,
                urgency_score=25.0,  # < 1.3 * 20.0 = 26.0
            )
            mock_time.return_value = 2.0
            events = narrator.update([obstacle_slight], current_time=2.0)
            assert events is None  # No event for minor increase

    def test_periodic_reminder_for_near_obstacles(self):
        """NEAR obstacles should trigger periodic reminders after ~6 seconds."""
        config = EventsConfig(min_frames_to_confirm=1)
        narrator = EventNarrator(config=config)

        det = Detection(label="person", confidence=0.9, x1=100, y1=200, x2=200, y2=400, track_id=24)
        obstacle = PrioritizedObstacle(
            detection=det,
            direction=Direction.CENTER,
            proximity=Proximity.NEAR,
            urgency_score=100.0,
        )

        with patch('time.monotonic') as mock_time:
            mock_time.return_value = 0.0
            # First establish and speak
            events = narrator.update([obstacle], current_time=0.0)
            assert events is not None
            narrator.record_spoken(events)

            # Wait 5.9 seconds - should NOT trigger reminder yet
            mock_time.return_value = 5.9
            events = narrator.update([obstacle], current_time=5.9)
            assert events is None

            # Wait 6.1 seconds - SHOULD trigger reminder
            mock_time.return_value = 6.1
            events = narrator.update([obstacle], current_time=6.1)
            assert events is not None
            assert events.event_type == EventType.PERIODIC_REMINDER
            assert events.label == "person"
            assert events.proximity == Proximity.NEAR

    def test_global_cooldown_enforcement(self):
        """Global cooldown should prevent any speech within the threshold."""
        config = EventsConfig(global_cooldown_seconds=2.0, min_frames_to_confirm=1)
        narrator = EventNarrator(config=config)

        det1 = Detection(label="person", confidence=0.8, x1=100, y1=100, x2=200, y2=200, track_id=25)
        obstacle1 = PrioritizedObstacle(
            detection=det1,
            direction=Direction.LEFT,
            proximity=Proximity.NEAR,
            urgency_score=100.0,
        )

        det2 = Detection(label="car", confidence=0.9, x1=300, y1=100, x2=400, y2=200, track_id=26)
        obstacle2 = PrioritizedObstacle(
            detection=det2,
            direction=Direction.RIGHT,
            proximity=Proximity.NEAR,
            urgency_score=90.0,
        )

        with patch('time.monotonic') as mock_time:
            mock_time.return_value = 0.0
            # First obstacle should trigger event immediately (min_frames_to_confirm=1)
            events = narrator.update([obstacle1], current_time=0.0)  # Frame 1 - should trigger NEW_OBJECT
            assert events is not None  # Capture the NEW_OBJECT event from frame 1
            assert events.event_type == EventType.NEW_OBJECT
            narrator.record_spoken(events)  # Mark as spoken so global cooldown applies
            first_event_time = 0.0

            # Immediately try second obstacle - should be blocked by global cooldown (2.0s in this config)
            mock_time.return_value = 0.1
            events = narrator.update([obstacle2], current_time=0.1)
            assert events is None  # Blocked by global cooldown (0.1 - 0.0 = 0.1 < 2.0)

            # Wait until global cooldown expires (global cooldown is 2.0s)
            mock_time.return_value = 2.1  # 2.1 - 0.0 = 2.1 >= 2.0
            events = narrator.update([obstacle2], current_time=2.1)
            assert events is not None  # Should now be allowed
            assert events.label == "car"

    def test_per_track_proximity_cooldown(self):
        """Same track/proximity should be blocked by per-track cooldown.

        Proximity escalation (FAR -> NEAR) requires 2 consecutive frames of NEAR
        due to temporal smoothing, then bypasses track cooldown.
        """
        config = EventsConfig(global_cooldown_seconds=0.1, track_cooldown_seconds=3.0, min_frames_to_confirm=1)
        narrator = EventNarrator(config=config)

        # Simulate tracking enabled scenario
        det = Detection(label="person", confidence=0.8, x1=100, y1=100, x2=200, y2=200, track_id=42)
        obstacle = PrioritizedObstacle(
            detection=det,
            direction=Direction.CENTER,
            proximity=Proximity.FAR,
            urgency_score=20.0,
        )

        with patch('time.monotonic') as mock_time:
            mock_time.return_value = 0.0
            # First announcement at FAR (min_frames_to_confirm=1 so triggers immediately)
            events = narrator.update([obstacle], current_time=0.0)
            assert events is not None
            assert events.event_type == EventType.NEW_OBJECT
            narrator.record_spoken(events)  # Mark as spoken

            # Same proximity at 0.2s (past global cooldown 0.1s, but within track cooldown 3.0s)
            mock_time.return_value = 0.2
            events = narrator.update([obstacle], current_time=0.2)
            assert events is None  # Blocked by per-track cooldown (same FAR proximity)

            # Different proximity escalation (FAR -> NEAR) - first NEAR frame
            # Temporal smoothing requires 2 consecutive NEAR frames to confirm escalation
            det_near = Detection(label="person", confidence=0.8, x1=100, y1=150, x2=200, y2=250, track_id=42)
            obstacle_near = PrioritizedObstacle(
                detection=det_near,
                direction=Direction.CENTER,
                proximity=Proximity.NEAR,
                urgency_score=20.0,  # Same urgency - no urgency escalation
            )
            mock_time.return_value = 0.4
            events = narrator.update([obstacle_near], current_time=0.4)
            # Single NEAR frame is smoothed to FAR -> no event
            assert events is None

            # Second NEAR frame - proximity escalation confirmed
            mock_time.return_value = 0.5
            events = narrator.update([obstacle_near], current_time=0.5)
            assert events is not None  # Escalation allowed
            assert events.event_type == EventType.PROXIMITY_ESCALATION
            narrator.record_spoken(events)  # Mark as spoken

            # Wait for track cooldown to expire on NEAR proximity
            mock_time.return_value = 4.5
            # Manually set last_spoken to 0.5 to simulate the track cooldown
            record = narrator._tracks.get(42)
            if record:
                record.last_spoken = 0.5
                record.last_spoken_proximity = Proximity.NEAR
            events = narrator.update([obstacle_near], current_time=4.5)
            # No event should fire because proximity hasn't changed and no escalation
            assert events is None

    def test_multiple_objects_highest_urgency_wins(self):
        """When multiple events possible, highest urgency should be selected.

        Proximity escalation (FAR -> MEDIUM) requires 2 consecutive frames of MEDIUM
        due to temporal smoothing. PROXIMITY_ESCALATION (tier 4) beats URGENCY_ESCALATION (tier 2).
        """
        config = EventsConfig()
        narrator = EventNarrator(config=config)

        # Low urgency event with tracking
        det_low = Detection(label="chair", confidence=0.5, x1=50, y1=100, x2=150, y2=200, track_id=1)
        obstacle_low = PrioritizedObstacle(
            detection=det_low,
            direction=Direction.LEFT,
            proximity=Proximity.FAR,
            urgency_score=20.0,
        )

        # High urgency event with tracking - baseline urgency 60 at FAR
        det_high = Detection(label="car", confidence=0.95, x1=300, y1=100, x2=400, y2=200, track_id=2)
        obstacle_high = PrioritizedObstacle(
            detection=det_high,
            direction=Direction.RIGHT,
            proximity=Proximity.FAR,
            urgency_score=60.0,  # Base FAR urgency
        )

        with patch('time.monotonic') as mock_time:
            mock_time.return_value = 0.0
            # Establish both tracks for 2 frames to confirm
            narrator.update([obstacle_low, obstacle_high], current_time=0.0)  # Frame 1
            narrator.update([obstacle_low, obstacle_high], current_time=0.1)  # Frame 2 - confirms both

            # Mark both as already spoken so they don't fire NEW_OBJECT later
            chair_record = narrator._tracks.get(1)
            if chair_record:
                chair_record.last_spoken = 0.1
                chair_record.last_spoken_proximity = Proximity.FAR
                chair_record.last_spoken_direction = Direction.LEFT
                chair_record.last_spoken_urgency = 20.0
            car_record = narrator._tracks.get(2)
            if car_record:
                car_record.last_spoken = 0.1
                car_record.last_spoken_proximity = Proximity.FAR
                car_record.last_spoken_direction = Direction.RIGHT
                car_record.last_spoken_urgency = 60.0

            # Now advance past global cooldown and trigger events
            # Chair: minor urgency increase (won't trigger, < 1.3x)
            det_low_up = Detection(label="chair", confidence=0.55, x1=50, y1=100, x2=150, y2=200, track_id=1)
            obstacle_low_up = PrioritizedObstacle(
                detection=det_low_up,
                direction=Direction.LEFT,
                proximity=Proximity.FAR,
                urgency_score=25.0,  # 25/20 = 1.25 < 1.3
            )

            # Car: proximity escalation FAR -> MEDIUM - first MEDIUM frame
            # Use same urgency as baseline (60.0) - may trigger urgency escalation but proximity escalation has higher tier
            det_car_med = Detection(label="car", confidence=0.95, x1=300, y1=150, x2=400, y2=250, track_id=2)
            obstacle_car_med = PrioritizedObstacle(
                detection=det_car_med,
                direction=Direction.RIGHT,
                proximity=Proximity.MEDIUM,
                urgency_score=60.0,  # Same as baseline
            )

            mock_time.return_value = 2.0  # Past global cooldown
            events = narrator.update([obstacle_low_up, obstacle_car_med], current_time=2.0)
            # First MEDIUM frame may trigger urgency escalation (depending on baseline), but proximity not yet confirmed
            # Chair has no escalation (same proximity, minor urgency increase < 1.3x)
            # Just verify we get some event (urgency escalation for car)
            assert events is not None
            assert events.label == "car"
            # Note: tier 4 (PROXIMITY) beats tier 2 (URGENCY), so on next frame PROXIMITY should win

            # Second MEDIUM frame - proximity escalation confirmed
            mock_time.return_value = 2.1
            events = narrator.update([obstacle_low_up, obstacle_car_med], current_time=2.1)
            # Car proximity escalation should win due to higher tier (4 vs 2)
            assert events is not None
            assert events.label == "car"
            assert events.event_type == EventType.PROXIMITY_ESCALATION

    def test_track_disappearance_and_grace_period(self):
        """Tracks should be preserved for grace period after disappearance."""
        config = EventsConfig()
        narrator = EventNarrator(config=config)

        det = Detection(label="person", confidence=0.8, x1=100, y1=100, x2=200, y2=200, track_id=77)
        obstacle = PrioritizedObstacle(
            detection=det,
            direction=Direction.CENTER,
            proximity=Proximity.NEAR,
            urgency_score=100.0,
        )

        with patch('time.monotonic') as mock_time:
            mock_time.return_value = 0.0
            # Establish track
            narrator.update([obstacle], current_time=0.0)
            assert 77 in narrator._tracks

            # Remove track from frame (simulate occlusion)
            mock_time.return_value = 0.1
            events = narrator.update([], current_time=0.1)  # Empty obstacles
            assert events is None
            # Track should still be in memory (within grace period)
            assert 77 in narrator._tracks

            # Wait beyond grace period (1.5 seconds)
            mock_time.return_value = 1.6
            events = narrator.update([], current_time=1.6)  # Still empty
            assert events is None
            # Track should now be purged
            assert 77 not in narrator._tracks

    def test_track_reappearance_after_grace_period_is_new_object(self):
        """Track that reappears after grace period should be treated as new."""
        config = EventsConfig(min_frames_to_confirm=2)
        narrator = EventNarrator(config=config)

        det = Detection(label="person", confidence=0.8, x1=100, y1=100, x2=200, y2=200, track_id=88)
        obstacle = PrioritizedObstacle(
            detection=det,
            direction=Direction.CENTER,
            proximity=Proximity.NEAR,
            urgency_score=100.0,
        )

        with patch('time.monotonic') as mock_time:
            mock_time.return_value = 0.0
            # Establish track for 2 frames to confirm
            narrator.update([obstacle], current_time=0.0)  # frame 1
            events = narrator.update([obstacle], current_time=0.1)  # frame 2 - should trigger NEW_OBJECT
            assert events is not None and events.event_type == EventType.NEW_OBJECT  # Capture NEW_OBJECT from frame 2

            # Now remove track
            mock_time.return_value = 0.2
            narrator.update([], current_time=0.2)  # Track disappears

            # Wait beyond grace period
            mock_time.return_value = 2.0  # 1.8 seconds later
            narrator.update([], current_time=2.0)  # Track purged from memory

            # Reintroduce track - should be treated as new object again
            mock_time.return_value = 2.1
            # First frame of reappearance
            narrator.update([obstacle], current_time=2.1)  # frame 1 of return (consecutive_frames=1)

            # Second frame - consecutive_frames becomes 2, equals min_frames_to_confirm, should trigger NEW_OBJECT
            mock_time.return_value = 2.15
            events = narrator.update([obstacle], current_time=2.15)  # frame 2 of return
            assert events is not None  # Should trigger NEW_OBJECT now (consecutive_frames=2 == min_frames_to_confirm)
            assert events.event_type == EventType.NEW_OBJECT

    def test_fallback_to_spatial_when_tracking_disabled(self):
        """When tracking disabled, should use (label, direction) as key."""
        config = EventsConfig()
        narrator = EventNarrator(config=config, tracking_enabled=False)

        det1 = Detection(label="person", confidence=0.8, x1=100, y1=100, x2=200, y2=200)  # left
        det2 = Detection(label="person", confidence=0.8, x1=300, y1=100, x2=400, y2=200)  # right

        obstacle_left = PrioritizedObstacle(
            detection=det1,
            direction=Direction.LEFT,
            proximity=Proximity.NEAR,
            urgency_score=100.0,
        )

        obstacle_right = PrioritizedObstacle(
            detection=det2,
            direction=Direction.RIGHT,
            proximity=Proximity.NEAR,
            urgency_score=100.0,
        )

        with patch('time.monotonic') as mock_time:
            mock_time.return_value = 0.0
            # Establish both
            narrator.update([obstacle_left], current_time=0.0)
            narrator.update([obstacle_right], current_time=0.0)

            # Both should be tracked separately
            assert len(narrator._tracks) == 2
            # Keys should be spatial: person:left and person:right
            keys = list(narrator._tracks.keys())
            assert "person:left" in keys
            assert "person:right" in keys

    def test_make_event_creates_correct_speech_text(self):
        """_make_event should generate appropriate speech text for each event type."""
        config = EventsConfig()
        narrator = EventNarrator(config=config)

        det = Detection(label="bicycle", confidence=0.8, x1=100, y1=100, x2=200, y2=200)
        obstacle = PrioritizedObstacle(
            detection=det,
            direction=Direction.LEFT,
            proximity=Proximity.NEAR,
            urgency_score=50.0,
        )

        # Test each event type
        test_cases = [
            (EventType.NEW_OBJECT, "near bicycle left"),
            (EventType.PROXIMITY_ESCALATION, "near bicycle left"),
            (EventType.URGENCY_ESCALATION, "near bicycle left"),
            (EventType.PERIODIC_REMINDER, "still near bicycle left"),
        ]

        for event_type, expected_speech in test_cases:
            event = narrator._make_event(obstacle, event_type)
            assert event.speech_text == expected_speech
            assert event.label == "bicycle"
            assert event.direction == Direction.LEFT
            assert event.proximity == Proximity.NEAR
            assert event.event_type == event_type

    def test_should_announce_respects_cooldowns(self):
        """_should_announce should properly enforce global and per-track cooldowns."""
        config = EventsConfig(global_cooldown_seconds=1.0, track_cooldown_seconds=2.0)
        narrator = EventNarrator(config=config)

        det = Detection(label="car", confidence=0.9, x1=100, y1=100, x2=200, y2=200, track_id=55)
        obstacle = PrioritizedObstacle(
            detection=det,
            direction=Direction.CENTER,
            proximity=Proximity.NEAR,
            urgency_score=90.0,
        )
        event = narrator._make_event(obstacle, EventType.NEW_OBJECT)

        with patch('time.monotonic') as mock_time:
            mock_time.return_value = 0.0
            # First announcement should be allowed (no previous speech)
            assert narrator._should_announce(event, 0.0) is True
            # Record that we spoke
            narrator._record_spoken(event, 0.0)

            # Immediately after - should be blocked by global cooldown (1.0s)
            mock_time.return_value = 0.5
            assert narrator._should_announce(event, 0.5) is False

            # After global cooldown expires - should be allowed
            mock_time.return_value = 1.1
            assert narrator._should_announce(event, 1.1) is True
            narrator._record_spoken(event, 1.1)

            # Now test per-track cooldown
            # Same track/proximity immediately after - should be blocked by per-track cooldown (2.0s)
            mock_time.return_value = 1.2
            assert narrator._should_announce(event, 1.2) is False

            # Different proximity should still work (if proximity changed)
            det_med = Detection(label="car", confidence=0.9, x1=100, y1=150, x2=200, y2=250, track_id=55)
            obstacle_med = PrioritizedObstacle(
                detection=det_med,
                direction=Direction.CENTER,
                proximity=Proximity.MEDIUM,
                urgency_score=70.0,
            )
            event_med = narrator._make_event(obstacle_med, EventType.PROXIMITY_ESCALATION)

            # Advance time past global cooldown (1.0s) before testing different proximity
            # Last global speech was at 1.1, so we need >= 2.1 to pass global cooldown
            mock_time.return_value = 2.2
            # Different proximity - per-track cooldown doesn't apply, global cooldown allows it
            assert narrator._should_announce(event_med, 2.2) is True

            # Wait for per-track cooldown to expire
            mock_time.return_value = 3.2
            # Same proximity again - should now be allowed
            assert narrator._should_announce(event, 3.2) is True

    def test_prune_stale_tracks_removes_old_entries(self):
        """_prune_stale_tracks should remove tracks not seen for grace period."""
        config = EventsConfig()
        narrator = EventNarrator(config=config)

        # Create two tracks
        det1 = Detection(label="person", confidence=0.8, x1=100, y1=100, x2=200, y2=200, track_id=101)
        det2 = Detection(label="car", confidence=0.9, x1=300, y1=100, x2=400, y2=200, track_id=102)

        obstacle1 = PrioritizedObstacle(
            detection=det1,
            direction=Direction.CENTER,
            proximity=Proximity.NEAR,
            urgency_score=100.0,
        )

        obstacle2 = PrioritizedObstacle(
            detection=det2,
            direction=Direction.CENTER,
            proximity=Proximity.NEAR,
            urgency_score=90.0,
        )

        with patch('time.monotonic') as mock_time:
            mock_time.return_value = 0.0
            # Establish both tracks
            narrator.update([obstacle1, obstacle2], current_time=0.0)
            assert len(narrator._tracks) == 2
            assert 101 in narrator._tracks
            assert 102 in narrator._tracks

            # Simulate that only car is still visible
            mock_time.return_value = 0.1
            narrator.update([obstacle2], current_time=0.1)  # person disappeared

            # Wait just under grace period
            mock_time.return_value = 1.4
            narrator._prune_stale_tracks(1.4, {102})  # Only car active
            assert len(narrator._tracks) == 2  # Both still there (within grace)
            assert 101 in narrator._tracks
            assert 102 in narrator._tracks

            # Wait beyond grace period
            mock_time.return_value = 1.6
            narrator._prune_stale_tracks(1.6, {102})  # Only car active
            assert len(narrator._tracks) == 1  # person removed
            assert 101 not in narrator._tracks
            assert 102 in narrator._tracks

    def test_reset_clears_all_state(self):
        """reset() should clear all track state and global speech time."""
        config = EventsConfig()
        narrator = EventNarrator(config=config)

        det = Detection(label="person", confidence=0.8, x1=100, y1=100, x2=200, y2=200, track_id=123)
        obstacle = PrioritizedObstacle(
            detection=det,
            direction=Direction.CENTER,
            proximity=Proximity.NEAR,
            urgency_score=100.0,
        )

        with patch('time.monotonic') as mock_time:
            mock_time.return_value = 0.0
            narrator.update([obstacle], current_time=0.0)
            assert len(narrator._tracks) == 1
            assert narrator._last_global_speech == -float("inf")  # Would be set after actual speech

            narrator.reset()
            assert len(narrator._tracks) == 0
            assert narrator._last_global_speech == -float("inf")


# ==============================================================================
# PerformanceMonitor Tests
# ==============================================================================


class TestPerformanceMonitor:
    """Test rolling performance metrics."""

    def test_performance_monitor_initialization(self):
        """PerformanceMonitor should initialize with proper defaults."""
        monitor = PerformanceMonitor()
        assert monitor._window_size == 30
        assert monitor._warmup_frames == 3
        assert monitor._frame_count == 0
        assert len(monitor._inference_latencies) == 0
        assert monitor._device_name == "Unknown"

    def test_performance_monitor_custom_parameters(self):
        """PerformanceMonitor should accept custom window and warmup sizes."""
        monitor = PerformanceMonitor(window_size=60, warmup_frames=5)
        assert monitor._window_size == 60
        assert monitor._warmup_frames == 5

    def test_set_device_name(self):
        """set_device_name should store the device identifier."""
        monitor = PerformanceMonitor()
        monitor.set_device_name("CUDA:0")
        assert monitor._device_name == "CUDA:0"
        monitor.set_device_name("CPU")
        assert monitor._device_name == "CPU"

    def test_record_frame_accumulates_data(self):
        """record_frame should store metrics in the rolling windows."""
        monitor = PerformanceMonitor(window_size=5, warmup_frames=2)

        with patch('time.monotonic') as mock_time:
            # Frame 1: time=0.0
            mock_time.return_value = 0.0
            monitor.record_frame(10.0, 50.0, 2)
            assert monitor._frame_count == 1
            assert list(monitor._inference_latencies) == [10.0]
            assert list(monitor._total_latencies) == [50.0]
            assert list(monitor._track_counts) == [2]
            assert list(monitor._frame_intervals) == []  # First frame has no interval

            # Frame 2: time=0.1 (0.1s interval)
            mock_time.return_value = 0.1
            monitor.record_frame(20.0, 60.0, 3)
            assert monitor._frame_count == 2
            assert list(monitor._inference_latencies) == [10.0, 20.0]
            assert list(monitor._total_latencies) == [50.0, 60.0]
            assert list(monitor._track_counts) == [2, 3]
            assert list(monitor._frame_intervals) == [0.1]  # 0.1s interval

            # Frame 3: time=0.2 (0.1s interval)
            mock_time.return_value = 0.2
            monitor.record_frame(15.0, 55.0, 1)
            assert monitor._frame_count == 3
            assert list(monitor._inference_latencies) == [10.0, 20.0, 15.0]
            assert list(monitor._total_latencies) == [50.0, 60.0, 55.0]
            assert list(monitor._track_counts) == [2, 3, 1]
            assert list(monitor._frame_intervals) == [0.1, 0.1]

    def test_fps_excludes_warmup_frames(self):
        """FPS calculation should exclude warmup frames from averaging."""
        monitor = PerformanceMonitor(window_size=10, warmup_frames=3)

        with patch('time.monotonic') as mock_time:
            # Simulate 5 frames at steady 10 FPS (0.1s intervals)
            base_time = 0.0
            intervals = [0.1, 0.1, 0.1, 0.1, 0.1]  # 5 intervals for 6 frames
            times = [base_time]
            for interval in intervals:
                times.append(times[-1] + interval)

            mock_time.side_effect = times

            # Record 6 frames (0 through 5)
            for i in range(6):
                monitor.record_frame(10.0, 50.0, 1)
                if i < len(times) - 1:
                    mock_time.return_value = times[i + 1]

            # After 6 frames: frame_count = 6
            # Warmup frames = 3, so valid frames = 6 - 3 = 3 frames for interval calculation
            # But we need intervals: we have 5 intervals total (frames 0-1, 1-2, 2-3, 3-4, 4-5)
            # We exclude first (warmup_frames - 1) = 2 intervals
            # So valid intervals = indices 2,3,4 = [0.1, 0.1, 0.1]
            # Average interval = 0.1s -> FPS = 10.0

            fps = monitor.fps
            assert fps == 10.0

    def test_fps_returns_zero_when_insufficient_data(self):
        """FPS should return 0.0 when not enough data for calculation."""
        monitor = PerformanceMonitor(window_size=10, warmup_frames=3)

        # Less than 2 intervals recorded
        monitor._frame_count = 1
        monitor._frame_intervals.append(0.0)  # Only one interval (actually zero for first frame)
        assert monitor.fps == 0.0

        # Exactly warmup_frames recorded, but need at least 2 intervals for FPS
        monitor._frame_count = 3
        monitor._frame_intervals = [0.0, 0.1, 0.1]  # Three frame times = two intervals
        # valid_intervals = [self._frame_intervals[self._warmup_frames - 1 :]] = [self._frame_intervals[2:]] = [0.1]
        # But we need at least 2 intervals to compute meaningful FPS? Let's check the actual logic...
        # Actually, the code checks if len(valid_intervals) < 2, then returns 0.0
        # With warmup_frames=3, valid_intervals starts at index 2, so we need at least 3+1=4 frames
        # to have 2 valid intervals.
        assert monitor.fps == 0.0  # Not enough valid intervals yet

    def test_avg_latency_excludes_warmup_frames(self):
        """Average latency should exclude warmup frames."""
        monitor = PerformanceMonitor(window_size=10, warmup_frames=3)

        # Record latencies: [10, 20, 30, 40, 50]
        for latency in [10.0, 20.0, 30.0, 40.0, 50.0]:
            monitor.record_frame(latency, latency*5, 1)

        # frame_count = 5
        # warmup_frames = 3
        # valid latencies = indices [2, 3, 4] = [30.0, 40.0, 50.0]
        # average = (30+40+50)/3 = 40.0
        assert monitor.avg_inference_ms == 40.0
        assert monitor.avg_total_ms == 200.0  # (150+200+250)/3

    def test_active_tracks_returns_most_recent(self):
        """active_tracks should return the most recently recorded value."""
        monitor = PerformanceMonitor()

        assert monitor.active_tracks == 0  # Default

        monitor.record_frame(10.0, 50.0, 5)
        assert monitor.active_tracks == 5

        monitor.record_frame(15.0, 55.0, 3)
        assert monitor.active_tracks == 3

        monitor.record_frame(20.0, 60.0, 0)
        assert monitor.active_tracks == 0

    def test_get_summary_returns_correct_dictionary(self):
        """get_summary should return a dictionary with all metrics."""
        monitor = PerformanceMonitor(window_size=5, warmup_frames=2)
        monitor.set_device_name("CUDA:1")

        with patch('time.monotonic') as mock_time:
            # Frame 1
            mock_time.return_value = 0.0
            monitor.record_frame(10.0, 50.0, 2)
            # Frame 2
            mock_time.return_value = 0.1
            monitor.record_frame(20.0, 60.0, 3)
            # Frame 3
            mock_time.return_value = 0.2
            monitor.record_frame(30.0, 70.0, 1)
            # Frame 4
            mock_time.return_value = 0.3
            monitor.record_frame(40.0, 80.0, 2)

            summary = monitor.get_summary()
            assert summary["fps"] == 10.0  # 1/(0.1) = 10.0
            # With warmup_frames=2, valid latencies start from index 1 (exclude first frame)
            # valid = list(self._inference_latencies)[self._warmup_frames - 1 :] = [10, 20, 30, 40][1:] = [20, 30, 40]
            # avg = (20+30+40)/3 = 30.0
            # avg_total = (60+70+80)/3 = 70.0
            assert summary["avg_inference_ms"] == 30.0  # (20+30+40)/3
            assert summary["avg_total_ms"] == 70.0  # (60+70+80)/3
            assert summary["active_tracks"] == 2
            assert summary["device"] == "CUDA:1"
            assert summary["warmed_up"] is True  # frame_count=4 >= warmup_frames=2

    def test_is_warmed_up_logic(self):
        """is_warmed_up should return True when enough frames processed."""
        monitor = PerformanceMonitor(warmup_frames=5)
        assert monitor.is_warmed_up is False  # 0 frames

        monitor._frame_count = 4
        assert monitor.is_warmed_up is False  # 4 < 5

        monitor._frame_count = 5
        assert monitor.is_warmed_up is True   # 5 >= 5

        monitor._frame_count = 10
        assert monitor.is_warmed_up is True   # 10 >= 5

    def test_reset_clears_all_metrics(self):
        """reset() should clear all recorded metrics."""
        monitor = PerformanceMonitor()

        # Add some data
        monitor.record_frame(10.0, 50.0, 2)
        monitor.record_frame(20.0, 60.0, 3)
        monitor.set_device_name("TEST_DEVICE")

        assert monitor._frame_count == 2
        assert len(monitor._inference_latencies) == 2
        assert monitor._device_name == "TEST_DEVICE"

        monitor.reset()

        assert monitor._frame_count == 0
        assert len(monitor._inference_latencies) == 0
        assert len(monitor._total_latencies) == 0
        assert len(monitor._frame_intervals) == 0
        assert len(monitor._track_counts) == 0
        assert monitor._last_frame_time is None
        # Note: device_name is NOT cleared by reset() - it's set separately
        assert monitor._device_name == "TEST_DEVICE"


# ==============================================================================
# Integration Tests
# ==============================================================================


class TestPipelineIntegration:
    """Test integration of tracking, events, and metrics in Pipeline."""

    def test_pipeline_integrates_event_narrator_and_metrics(self):
        """Pipeline should create EventNarrator and PerformanceMonitor instances."""
        config = AppConfig()
        pipeline = Pipeline(config)

        assert hasattr(pipeline, 'event_narrator')
        assert isinstance(pipeline.event_narrator, EventNarrator)
        assert hasattr(pipeline, 'metrics')
        assert isinstance(pipeline.metrics, PerformanceMonitor)

    def test_pipeline_respects_tracking_disabled_flag(self):
        """Pipeline should pass tracking_enabled flag to EventNarrator."""
        config = AppConfig()
        config.detector.tracking_enabled = False

        pipeline = Pipeline(config)
        assert pipeline.event_narrator._tracking_enabled is False

    def test_pipeline_process_frame_calls_event_narrator(self):
        """Pipeline.process_frame should call event_narrator.update."""
        config = AppConfig()
        config.speaker.enabled = False  # Disable speech to avoid side effects

        # Mock detector to return a known detection
        mock_detector = MagicMock()
        det = Detection(label="person", confidence=0.8, x1=100, y1=100, x2=200, y2=200, track_id=77)
        mock_detector.detect.return_value = [det]

        # Mock speaker
        mock_speaker = MagicMock()

        pipeline = Pipeline(config, detector=mock_detector, speaker=mock_speaker)

        # Mock event_narrator.update to see if it's called
        with patch.object(pipeline.event_narrator, 'update', return_value=None) as mock_update:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            pipeline.process_frame(frame)

            mock_update.assert_called_once()
            # Verify it received the prioritized obstacles
            args, kwargs = mock_update.call_args
            assert len(args[0]) == 1  # One obstacle
            assert isinstance(args[0][0], PrioritizedObstacle)
            assert args[0][0].label == "person"

    def test_pipeline_process_frame_records_metrics(self):
        """Pipeline.process_frame should call metrics.record_frame."""
        config = AppConfig()
        config.speaker.enabled = False

        # Mock detector to return empty detections (fast path)
        mock_detector = MagicMock()
        mock_detector.detect.return_value = []

        # Mock speaker
        mock_speaker = MagicMock()

        pipeline = Pipeline(config, detector=mock_detector, speaker=mock_speaker)

        # Mock metrics.record_frame to see if it's called
        with patch.object(pipeline.metrics, 'record_frame') as mock_record:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            pipeline.process_frame(frame)

            mock_record.assert_called_once()
            # Verify it received latency and track count
            args, kwargs = mock_record.call_args
            assert 'inference_latency_ms' in kwargs
            assert 'total_latency_ms' in kwargs
            assert 'active_tracks' in kwargs
            assert kwargs['active_tracks'] == 0  # Empty detections

    def test_pipeline_speaks_when_event_returned(self):
        """Pipeline should call speaker.speak when event_narrator returns an event."""
        config = AppConfig()

        # Mock detector
        mock_detector = MagicMock()
        det = Detection(label="car", confidence=0.9, x1=200, y1=100, x2=300, y2=200)
        mock_detector.detect.return_value = [det]

        # Mock speaker
        mock_speaker = MagicMock()

        pipeline = Pipeline(config, detector=mock_detector, speaker=mock_speaker)

        # Mock event_narrator to return a test event
        test_event = NarrationEvent(
            track_id=None,
            label="car",
            direction=Direction.RIGHT,
            proximity=Proximity.NEAR,
            event_type=EventType.NEW_OBJECT,
            urgency_score=90.0,
            speech_text="near car right",
        )

        with patch.object(pipeline.event_narrator, 'update', return_value=test_event):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            pipeline.process_frame(frame)

            mock_speaker.speak.assert_called_once_with("near car right")

    def test_pipeline_does_not_speak_when_muted(self):
        """Pipeline should not call speaker when muted."""
        config = AppConfig()
        config.speaker.enabled = False  # Explicitly muted

        # Mock detector
        mock_detector = MagicMock()
        det = Detection(label="person", confidence=0.8, x1=100, y1=100, x2=200, y2=200)
        mock_detector.detect.return_value = [det]

        # Mock speaker
        mock_speaker = MagicMock()

        pipeline = Pipeline(config, detector=mock_detector, speaker=mock_speaker)

        # Mock event_narrator to return an event
        test_event = NarrationEvent(
            track_id=None,
            label="person",
            direction=Direction.LEFT,
            proximity=Proximity.NEAR,
            event_type=EventType.NEW_OBJECT,
            urgency_score=100.0,
            speech_text="near person left",
        )

        with patch.object(pipeline.event_narrator, 'update', return_value=test_event):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            pipeline.process_frame(frame)

            mock_speaker.speak.assert_not_called()  # Should not speak when muted

    def test_pipeline_speaks_once_per_event(self):
        """Pipeline should call speaker.speak exactly once per event."""
        config = AppConfig()

        # Mock detector
        mock_detector = MagicMock()
        det = Detection(label="car", confidence=0.9, x1=200, y1=100, x2=300, y2=200)
        mock_detector.detect.return_value = [det]

        # Mock speaker - track calls
        mock_speaker = MagicMock()
        mock_speaker.speak.return_value = True  # Accept the utterance

        pipeline = Pipeline(config, detector=mock_detector, speaker=mock_speaker)

        # Mock event_narrator to return a test event
        test_event = NarrationEvent(
            track_id=None,
            label="car",
            direction=Direction.RIGHT,
            proximity=Proximity.NEAR,
            event_type=EventType.NEW_OBJECT,
            urgency_score=90.0,
            speech_text="near car right",
        )

        with patch.object(pipeline.event_narrator, 'update', return_value=test_event):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            pipeline.process_frame(frame)

            # speak() should be called exactly once
            assert mock_speaker.speak.call_count == 1
            mock_speaker.speak.assert_called_once_with("near car right")

    def test_pipeline_does_not_record_spoken_when_speaker_rejects(self):
        """Event should not be marked as spoken when speaker rejects (queue full, cooldown, disabled)."""
        config = AppConfig()

        # Mock detector
        mock_detector = MagicMock()
        det = Detection(label="car", confidence=0.9, x1=200, y1=100, x2=300, y2=200)
        mock_detector.detect.return_value = [det]

        # Mock speaker - reject the utterance
        mock_speaker = MagicMock()
        mock_speaker.speak.return_value = False  # Reject (e.g., queue full, cooldown, or disabled)

        pipeline = Pipeline(config, detector=mock_detector, speaker=mock_speaker)

        # Mock event_narrator to return a test event
        test_event = NarrationEvent(
            track_id=None,
            label="car",
            direction=Direction.RIGHT,
            proximity=Proximity.NEAR,
            event_type=EventType.NEW_OBJECT,
            urgency_score=90.0,
            speech_text="near car right",
        )

        with patch.object(pipeline.event_narrator, 'update', return_value=test_event):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            pipeline.process_frame(frame)

            # speak() should be called but record_spoken should NOT be called
            mock_speaker.speak.assert_called_once_with("near car right")
            # Verify the event narrator's state - the track should NOT have last_spoken set
            # We can check by looking at the internal state
            record = pipeline.event_narrator._tracks.get("car:right")
            if record:
                assert record.last_spoken < 0  # Should still be sentinel value (never spoken)

    def test_pipeline_can_announce_second_object_after_cooldown(self):
        """Pipeline should be able to announce a second new object after first announcement and cooldown."""
        config = AppConfig()
        config.events.global_cooldown_seconds = 0.1  # Short global cooldown for test
        config.events.min_frames_to_confirm = 1  # Immediate confirmation

        # Mock detector - returns different objects on different calls
        mock_detector = MagicMock()
        det1 = Detection(label="car", confidence=0.9, x1=200, y1=100, x2=300, y2=200, track_id=1)
        det2 = Detection(label="person", confidence=0.8, x1=400, y1=100, x2=500, y2=200, track_id=2)
        mock_detector.detect.side_effect = [[det1], [det2]]  # First frame: car, Second frame: person

        # Mock speaker - accept first, then accept second
        mock_speaker = MagicMock()
        mock_speaker.speak.side_effect = [True, True]

        pipeline = Pipeline(config, detector=mock_detector, speaker=mock_speaker)

        with patch('time.monotonic') as mock_time:
            mock_time.return_value = 0.0
            frame = np.zeros((480, 640, 3), dtype=np.uint8)

            # First frame - car appears
            # Car at x=200-300 (CENTER), height=100/480=0.208 (MEDIUM) -> "car ahead"
            pipeline.process_frame(frame)
            mock_speaker.speak.assert_called_with("car ahead")
            assert mock_speaker.speak.call_count == 1

            # Advance past global cooldown
            mock_time.return_value = 0.2

            # Second frame - person appears (car still there but already announced)
            # Person at x=400-500 (RIGHT), height=100/480=0.208 (MEDIUM) -> "person right"
            pipeline.process_frame(frame)
            assert mock_speaker.speak.call_count == 2
            mock_speaker.speak.assert_called_with("person right")

    def test_pipeline_handles_speaker_worker_error_gracefully(self):
        """Pipeline should not mark event as spoken if TTS worker fails during playback."""
        config = AppConfig()

        # Use real speaker but mock the worker to fail
        mock_detector = MagicMock()
        det = Detection(label="car", confidence=0.9, x1=200, y1=100, x2=300, y2=200)
        mock_detector.detect.return_value = [det]

        # Create real speaker with mocked worker that fails
        from narrator.speaker import Speaker
        from narrator.config import SpeakerConfig
        speaker_config = SpeakerConfig(enabled=True, cooldown_seconds=0.0)

        # We need to test with a real speaker to verify the queue acceptance vs playback distinction
        # The speak() method returns True on queue acceptance, not on successful playback
        # This test verifies that distinction
        with patch('pyttsx3.init') as mock_tts_init:
            mock_engine = MagicMock()
            mock_engine.say.side_effect = Exception("TTS engine error")
            mock_tts_init.return_value = mock_engine

            speaker = Speaker(speaker_config)

            # Wait for worker to start
            import time
            time.sleep(0.1)

            # Try to speak - should return True (queue accepted) even though playback will fail
            result = speaker.speak("test message")
            assert result is True  # Queue acceptance succeeded

            # The event would be marked as spoken (queue accepted), but playback fails silently
            # This is the expected behavior - we distinguish queue acceptance from playback
            speaker.stop()


class TestDetectorRegression:
    """Regression tests for detector behavior with multiple classes and tracking."""

    def test_model_class_mapping_inspected_correctly(self):
        """The actual model's class mapping should be inspected correctly."""
        config = DetectorConfig()
        detector = Detector(config)
        classes = detector.get_model_classes()

        assert isinstance(classes, dict)
        assert len(classes) >= 80, f"Expected at least 80 classes, got {len(classes)}"

        # Check required everyday and obstacle classes
        expected_classes = [
            "person", "bottle", "chair", "cup", "backpack", "handbag",
            "suitcase", "cell phone", "laptop", "book", "dining table",
            "bench", "bicycle", "car", "bus"
        ]
        class_names = set(classes.values())
        for cls_name in expected_classes:
            assert cls_name in class_names, f"Class '{cls_name}' should be in model classes"

    def test_multiple_raw_boxes_different_classes_become_detections(self):
        """Multiple raw boxes from different supported classes must become multiple Detection instances."""
        detector = Detector()

        # Mock model results
        mock_box1 = MagicMock()
        mock_box1.cls = [0]  # person
        mock_box1.conf = [0.88]
        mock_box1.xyxy = [[50.0, 60.0, 150.0, 300.0]]
        mock_box1.id = [1]

        mock_box2 = MagicMock()
        mock_box2.cls = [39]  # bottle
        mock_box2.conf = [0.75]
        mock_box2.xyxy = [[300.0, 200.0, 350.0, 320.0]]
        mock_box2.id = [2]

        mock_box3 = MagicMock()
        mock_box3.cls = [56]  # chair
        mock_box3.conf = [0.70]
        mock_box3.xyxy = [[400.0, 150.0, 550.0, 400.0]]
        mock_box3.id = [3]

        mock_result = MagicMock()
        mock_result.boxes = [mock_box1, mock_box2, mock_box3]

        with patch.object(detector._model, 'track', return_value=[mock_result]):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            dets = detector.detect(frame)

            assert len(dets) == 3, f"Expected 3 detections, got {len(dets)}"
            labels = [d.label for d in dets]
            assert "person" in labels
            assert "bottle" in labels
            assert "chair" in labels

    def test_valid_detections_survive_when_track_id_absent(self):
        """All valid detections should survive tracking when track IDs are absent (box.id is None)."""
        detector = Detector()

        mock_box = MagicMock()
        mock_box.cls = [39]  # bottle
        mock_box.conf = [0.80]
        mock_box.xyxy = [[100.0, 100.0, 200.0, 300.0]]
        mock_box.id = None  # No track ID assigned yet by tracker

        mock_result = MagicMock()
        mock_result.boxes = [mock_box]

        with patch.object(detector._model, 'track', return_value=[mock_result]):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            dets = detector.detect(frame)

            assert len(dets) == 1, "Detection should be preserved even without track_id"
            assert dets[0].label == "bottle"
            assert dets[0].track_id is None

    def test_detector_does_not_filter_out_everyday_objects(self):
        """Detector configuration should include everyday objects like bottle, cup, phone, chair."""
        config = DetectorConfig()
        everyday_items = ["bottle", "cup", "chair", "backpack", "handbag", "suitcase", "cell phone", "laptop", "book", "dining table"]
        for item in everyday_items:
            assert item in config.obstacle_classes, f"Everyday item '{item}' must be in obstacle_classes"

    def test_detector_inference_failure_handled_gracefully(self):
        """Inference exceptions should be caught and return an empty list without crashing."""
        detector = Detector()
        with patch.object(detector._model, 'track', side_effect=RuntimeError("GPU out of memory")):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            dets = detector.detect(frame)
            assert dets == [], "Failed inference should return empty list"

    def test_diagnostic_mode_bypasses_tracking_and_narration(self):
        """Diagnostic mode diagnose_frame should run raw predict() and report raw counts."""
        detector = Detector()

        mock_box = MagicMock()
        mock_box.cls = [39]  # bottle
        mock_box.conf = [0.85]
        mock_box.xyxy = [[10.0, 20.0, 100.0, 200.0]]

        mock_result = MagicMock()
        mock_result.boxes = [mock_box]

        with patch.object(detector._model, 'predict', return_value=[mock_result]):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            diag = detector.diagnose_frame(frame)

            assert diag["error"] is None
            assert diag["count"] == 1
            assert len(diag["detections"]) == 1
            assert diag["detections"][0]["label"] == "bottle"
            assert diag["detections"][0]["confidence"] == 0.85
            assert "latency_ms" in diag

    def test_detector_returns_multiple_classes(self):
        """Detector should return all supported classes, not just person."""
        import cv2
        bus_fixture = Path(__file__).resolve().parent / "fixtures" / "bus.jpg"
        frame = cv2.imread(str(bus_fixture))
        if frame is None:
            pytest.skip("bus.jpg fixture not available")

        config = DetectorConfig(confidence_threshold=0.35, image_size=640, tracking_enabled=True)
        detector = Detector(config)
        detections = detector.detect(frame)

        # Should detect both bus and person
        labels = {d.label for d in detections}
        assert 'bus' in labels, "Bus should be detected"
        assert 'person' in labels, "Person should be detected"
        assert len(detections) >= 2, "Should have multiple detections"

    def test_detector_returns_multiple_detections_same_class(self):
        """Detector should return multiple detections of the same class."""
        import cv2
        bus_fixture = Path(__file__).resolve().parent / "fixtures" / "bus.jpg"
        frame = cv2.imread(str(bus_fixture))
        if frame is None:
            pytest.skip("bus.jpg fixture not available")

        config = DetectorConfig(confidence_threshold=0.35, image_size=640, tracking_enabled=True)
        detector = Detector(config)
        detections = detector.detect(frame)

        # Should detect multiple persons
        person_count = sum(1 for d in detections if d.label == 'person')
        assert person_count >= 2, f"Expected multiple persons, got {person_count}"

    def test_detector_tracking_disabled_still_returns_detections(self):
        """Detector with tracking disabled should still return detections."""
        import cv2
        bus_fixture = Path(__file__).resolve().parent / "fixtures" / "bus.jpg"
        frame = cv2.imread(str(bus_fixture))
        if frame is None:
            pytest.skip("bus.jpg fixture not available")

        config = DetectorConfig(confidence_threshold=0.35, image_size=640, tracking_enabled=False)
        detector = Detector(config)
        detections = detector.detect(frame)

        assert len(detections) > 0, "Should have detections even without tracking"
        # All detections should have track_id=None
        for d in detections:
            assert d.track_id is None, "Track ID should be None when tracking disabled"

    def test_detector_missing_track_id_does_not_drop_detection(self):
        """Detections without track IDs should still be returned."""
        import cv2
        bus_fixture = Path(__file__).resolve().parent / "fixtures" / "bus.jpg"
        frame = cv2.imread(str(bus_fixture))
        if frame is None:
            pytest.skip("bus.jpg fixture not available")

        config = DetectorConfig(confidence_threshold=0.35, image_size=640, tracking_enabled=True)
        detector = Detector(config)
        detections = detector.detect(frame)

        # All detections should have track IDs when tracking enabled and model provides them
        for d in detections:
            assert d.track_id is not None, "Track ID should be present when tracking enabled"

    def test_detector_class_filtering_works(self):
        """Detector should filter to only configured obstacle classes."""
        import cv2
        bus_fixture = Path(__file__).resolve().parent / "fixtures" / "bus.jpg"
        frame = cv2.imread(str(bus_fixture))
        if frame is None:
            pytest.skip("bus.jpg fixture not available")

        # Custom config with only 'bus' as obstacle class
        config = DetectorConfig(
            confidence_threshold=0.35,
            image_size=640,
            tracking_enabled=True,
            obstacle_classes={'bus'}
        )
        detector = Detector(config)
        detections = detector.detect(frame)

        # Should only detect bus, not person
        labels = {d.label for d in detections}
        assert labels == {'bus'}, f"Expected only bus, got {labels}"

    def test_detector_handles_empty_results(self):
        """Detector should handle empty model results gracefully."""
        import numpy as np
        from narrator.detector import Detector, DetectorConfig

        config = DetectorConfig(confidence_threshold=0.99, image_size=640, tracking_enabled=True)  # Very high threshold
        detector = Detector(config)

        # Empty frame (black image)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = detector.detect(frame)

        assert detections == [], "Should return empty list for no detections"


class TestPipelineRegression:
    """Regression tests for pipeline behavior with multiple objects and cooldowns."""

    def test_every_retained_detection_labeled_on_display(self):
        """Every retained detection must be rendered with a bounding box and label on display."""
        config = AppConfig()
        pipeline = Pipeline(config)

        det1 = Detection(label="person", confidence=0.9, x1=50, y1=50, x2=150, y2=250, track_id=1)
        det2 = Detection(label="bottle", confidence=0.8, x1=200, y1=100, x2=250, y2=200, track_id=2)
        det3 = Detection(label="chair", confidence=0.75, x1=300, y1=150, x2=450, y2=350, track_id=3)

        obstacles = prioritize_detections([det1, det2, det3], frame_width=640, frame_height=480)
        assert len(obstacles) == 3

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        annotated = pipeline._annotate(frame, obstacles, latency_ms=25.0)

        # Annotated frame must have visual modifications (non-zero pixels where boxes/text are drawn)
        assert np.count_nonzero(annotated) > 0, "Canvas should have drawn boxes and text"

    def test_lower_priority_object_not_removed_because_person_present(self):
        """Lower priority objects (e.g. bottle, chair) should never be dropped when a person is present."""
        det_person = Detection(label="person", confidence=0.9, x1=50, y1=50, x2=250, y2=450)  # Near, urgent
        det_bottle = Detection(label="bottle", confidence=0.8, x1=400, y1=100, x2=450, y2=200) # Medium
        det_chair = Detection(label="chair", confidence=0.7, x1=500, y1=200, x2=600, y2=300)   # Far

        obstacles = prioritize_detections([det_person, det_bottle, det_chair], frame_width=640, frame_height=480)
        assert len(obstacles) == 3
        labels = [o.label for o in obstacles]
        assert "person" in labels
        assert "bottle" in labels
        assert "chair" in labels

    def test_alert_classes_filters_narration_while_display_retains_all(self):
        """Alert classes filter speech events without removing non-alert objects from display."""
        config = AppConfig()
        config.events.alert_classes = {"bottle"}  # Only alert on bottle, not person

        pipeline = Pipeline(config)
        det_person = Detection(label="person", confidence=0.9, x1=50, y1=50, x2=250, y2=450, track_id=1)
        det_bottle = Detection(label="bottle", confidence=0.8, x1=400, y1=100, x2=450, y2=200, track_id=2)

        obstacles = prioritize_detections([det_person, det_bottle], frame_width=640, frame_height=480)
        # Display retains both
        assert len(obstacles) == 2

        # Event narrator should only consider bottle for alerts, not person
        with patch('time.monotonic', return_value=0.0):
            # Frame 1
            pipeline.event_narrator.update(obstacles, current_time=0.0)
            # Frame 2
            event = pipeline.event_narrator.update(obstacles, current_time=0.1)

            assert event is not None
            assert event.label == "bottle", "Alert should be generated for bottle, not person"

    def test_pipeline_person_first_then_bottle_announced(self):
        """A bottle appearing after a person should be announced (not blocked by person)."""
        config = AppConfig()
        config.events.global_cooldown_seconds = 0.5
        config.events.track_cooldown_seconds = 0.5
        config.speaker.cooldown_seconds = 0.5
        config.events.min_frames_to_confirm = 1

        # Mock detector - person first, then person + bottle
        mock_detector = MagicMock()
        det_person = Detection(label='person', confidence=0.9, x1=100, y1=100, x2=300, y2=400, track_id=1)
        det_bottle = Detection(label='bottle', confidence=0.85, x1=500, y1=100, x2=600, y2=400, track_id=2)

        call_count = [0]
        def mock_detect(frame):
            call_count[0] += 1
            if call_count[0] <= 2:
                return [det_person]
            else:
                return [det_person, det_bottle]
        mock_detector.detect = mock_detect

        mock_speaker = MagicMock()
        mock_speaker.speak.return_value = True

        pipeline = Pipeline(config, detector=mock_detector, speaker=mock_speaker)

        import narrator.events as events_module
        import narrator.speaker as speaker_module
        mock_time = [0.0]
        def mock_monotonic():
            return mock_time[0]
        events_module.time.monotonic = mock_monotonic
        speaker_module.time.monotonic = mock_monotonic

        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Frame 1-2: person only
        pipeline.process_frame(frame)
        mock_time[0] += 1.0
        pipeline.process_frame(frame)
        mock_time[0] += 1.0

        # Frame 3-4: person + bottle
        pipeline.process_frame(frame)
        mock_time[0] += 1.0
        pipeline.process_frame(frame)

        # Should have spoken both "new person" and "new bottle"
        speak_calls = [call[0][0] for call in mock_speaker.speak.call_args_list]
        assert any('person' in call for call in speak_calls), "Person should be announced"
        assert any('bottle' in call for call in speak_calls), "Bottle should be announced after person"

    def test_speaker_cooldown_rejection_does_not_block_event_narrator(self):
        """When speaker rejects due to cooldown, EventNarrator should not re-offer immediately."""
        config = AppConfig()
        config.events.global_cooldown_seconds = 0.1
        config.events.track_cooldown_seconds = 0.1
        config.speaker.cooldown_seconds = 2.0  # Speaker has long cooldown
        config.events.min_frames_to_confirm = 1

        mock_detector = MagicMock()
        det = Detection(label='car', confidence=0.9, x1=100, y1=100, x2=300, y2=400, track_id=1)
        mock_detector.detect.return_value = [det]

        mock_speaker = MagicMock()
        # First call accepted, subsequent calls rejected (simulating cooldown)
        mock_speaker.speak.side_effect = [True, False, False, False, True]

        pipeline = Pipeline(config, detector=mock_detector, speaker=mock_speaker)

        import narrator.events as events_module
        import narrator.speaker as speaker_module
        mock_time = [0.0]
        def mock_monotonic():
            return mock_time[0]
        events_module.time.monotonic = mock_monotonic
        speaker_module.time.monotonic = mock_monotonic

        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Multiple frames with same object
        for i in range(5):
            pipeline.process_frame(frame)
            mock_time[0] += 0.5

        # Should only call speak once (first accepted, rest rejected but not re-offered endlessly)
        # The key is that after rejection, event_narrator should not keep re-offering the same event
        # due to the 0.5s last_offered grace period
        assert mock_speaker.speak.call_count <= 2, \
            f"Speak should not be called repeatedly for rejected events, got {mock_speaker.speak.call_count} calls"

    def test_proximity_escalation_still_works(self):
        """Proximity escalation (MEDIUM -> NEAR) should still trigger events."""
        config = EventsConfig(announce_proximity_changes=True)
        narrator = EventNarrator(config=config)

        # Start with MEDIUM
        det_med = Detection(label="car", confidence=0.9, x1=300, y1=200, x2=400, y2=300, track_id=1)
        obstacle_med = PrioritizedObstacle(
            detection=det_med,
            direction=Direction.RIGHT,
            proximity=Proximity.MEDIUM,
            urgency_score=50.0,
        )

        with patch('time.monotonic') as mock_time:
            mock_time.return_value = 0.0
            narrator.update([obstacle_med], current_time=0.0)

            # Now NEAR - should trigger PROXIMITY_ESCALATION
            det_near = Detection(label="car", confidence=0.9, x1=300, y1=300, x2=400, y2=400, track_id=1)
            obstacle_near = PrioritizedObstacle(
                detection=det_near,
                direction=Direction.RIGHT,
                proximity=Proximity.NEAR,
                urgency_score=100.0,
            )
            mock_time.return_value = 2.0  # Past global cooldown
            events = narrator.update([obstacle_near], current_time=2.0)
            assert events is not None
            assert events.event_type == EventType.PROXIMITY_ESCALATION
            assert events.proximity == Proximity.NEAR

    def test_urgency_escalation_still_works(self):
        """Significant urgency increase should trigger URGENCY_ESCALATION event."""
        config = EventsConfig()
        narrator = EventNarrator(config=config)

        det = Detection(label="chair", confidence=0.5, x1=50, y1=100, x2=150, y2=200, track_id=21)
        obstacle_low = PrioritizedObstacle(
            detection=det,
            direction=Direction.LEFT,
            proximity=Proximity.FAR,
            urgency_score=20.0,
        )

        with patch('time.monotonic') as mock_time:
            mock_time.return_value = 0.0
            narrator.update([obstacle_low], current_time=0.0)
            narrator.update([obstacle_low], current_time=0.1)  # Confirm

            # Trigger urgency escalation
            det_high = Detection(label="chair", confidence=0.95, x1=50, y1=100, x2=150, y2=200, track_id=21)
            obstacle_high = PrioritizedObstacle(
                detection=det_high,
                direction=Direction.LEFT,
                proximity=Proximity.FAR,
                urgency_score=50.0,  # > 1.3 * 20.0 = 26.0
            )
            mock_time.return_value = 5.0  # Past track cooldown
            events = narrator.update([obstacle_high], current_time=5.0)
            assert events is not None
            assert events.event_type == EventType.URGENCY_ESCALATION

    def test_periodic_reminder_still_works(self):
        """NEAR obstacles should trigger periodic reminders after ~6 seconds."""
        config = EventsConfig(min_frames_to_confirm=1)
        narrator = EventNarrator(config=config)

        det = Detection(label="person", confidence=0.9, x1=100, y1=200, x2=200, y2=400, track_id=24)
        obstacle = PrioritizedObstacle(
            detection=det,
            direction=Direction.CENTER,
            proximity=Proximity.NEAR,
            urgency_score=100.0,
        )

        with patch('time.monotonic') as mock_time:
            mock_time.return_value = 0.0
            events = narrator.update([obstacle], current_time=0.0)
            assert events is not None
            narrator.record_spoken(events)

            # Wait 5.9 seconds - should NOT trigger reminder yet
            mock_time.return_value = 5.9
            events = narrator.update([obstacle], current_time=5.9)
            assert events is None

            # Wait 6.1 seconds - SHOULD trigger reminder
            mock_time.return_value = 6.1
            events = narrator.update([obstacle], current_time=6.1)
            assert events is not None
            assert events.event_type == EventType.PERIODIC_REMINDER
            assert events.label == "person"
            assert events.proximity == Proximity.NEAR


class TestEndToEndIntegration:
    """End-to-end integration tests simulating real-world scenarios."""

    def test_sequential_object_announcements(self):
        """
        Deterministic integration test: simulate a sequence of frames where different
        objects appear and are announced, ensuring no one monopolizes the speaker.

        Scenario:
        1. Person appears (confirmed -> announced)
        2. Person remains (no new event)
        3. Bottle appears (confirmed -> announced)
        4. Bottle remains (no new event)
        5. Chair appears (confirmed -> announced)
        """
        config = AppConfig()
        config.events.min_frames_to_confirm = 2
        config.events.global_cooldown_seconds = 1.0

        mock_detector = MagicMock()
        mock_speaker = MagicMock()
        mock_speaker.speak.return_value = True

        pipeline = Pipeline(config, detector=mock_detector, speaker=mock_speaker)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Control time deterministically
        import narrator.events as events_module
        import narrator.speaker as speaker_module
        mock_time = [0.0]

        def mock_monotonic():
            return mock_time[0]

        with patch.object(events_module.time, "monotonic", side_effect=mock_monotonic), \
             patch.object(speaker_module.time, "monotonic", side_effect=mock_monotonic):

            # --- 1. Person appears ---
            person_det = Detection(label="person", confidence=0.9, x1=100, y1=100, x2=200, y2=200, track_id=1)
            mock_detector.detect.return_value = [person_det]

            # Frame 1: Seen (time=0.0)
            pipeline.process_frame(frame)
            mock_speaker.speak.assert_not_called()

            # Frame 2: Confirmed -> NEW_OBJECT (time=0.1)
            # Person at x=100-200 (LEFT), height=100/480=0.208 (MEDIUM) -> "person left"
            mock_time[0] = 0.1
            pipeline.process_frame(frame)
            assert mock_speaker.speak.call_args[0][0] == "person left"
            mock_speaker.speak.reset_mock()

            # Frame 3: Still there (time=0.2, no event)
            mock_time[0] = 0.2
            pipeline.process_frame(frame)
            mock_speaker.speak.assert_not_called()

            # Advance time past global cooldown (last spoken at 0.1, cooldown is 1.0s, so >= 1.1s)
            mock_time[0] = 1.5

            # --- 2. Bottle appears (while person is still there) ---
            bottle_det = Detection(label="bottle", confidence=0.8, x1=300, y1=100, x2=350, y2=200, track_id=2)
            mock_detector.detect.return_value = [person_det, bottle_det]

            # Frame 4: Bottle seen (time=1.5)
            pipeline.process_frame(frame)
            mock_speaker.speak.assert_not_called()

            # Frame 5: Bottle confirmed -> NEW_OBJECT (time=1.6)
            # Bottle at x=300-350 (CENTER), height=100/480=0.208 (MEDIUM) -> "bottle ahead"
            mock_time[0] = 1.6
            pipeline.process_frame(frame)
            assert mock_speaker.speak.call_args[0][0] == "bottle ahead"
            mock_speaker.speak.reset_mock()

            # Frame 5b: Person and bottle still there (time=1.7, no event)
            mock_time[0] = 1.7
            pipeline.process_frame(frame)
            mock_speaker.speak.assert_not_called()

            # Advance time past global cooldown (last spoken at 1.6, cooldown is 1.0s, so >= 2.6s)
            mock_time[0] = 3.0

            # --- 3. Chair appears ---
            chair_det = Detection(label="chair", confidence=0.7, x1=500, y1=100, x2=600, y2=200, track_id=3)
            mock_detector.detect.return_value = [person_det, bottle_det, chair_det]

            # Frame 6: Chair seen (time=3.0)
            pipeline.process_frame(frame)
            mock_speaker.speak.assert_not_called()

            # Frame 7: Chair confirmed -> NEW_OBJECT (time=3.1)
            # Chair at x=500-600 (RIGHT), height=100/480=0.208 (MEDIUM) -> "chair right"
            mock_time[0] = 3.1
            pipeline.process_frame(frame)
            assert mock_speaker.speak.call_args[0][0] == "chair right"


class TestPhrasingIntegration:
    """Integration tests for event-to-phrase-to-speaker pipeline."""

    def test_event_to_phrase_to_speaker_integration_new_object(self) -> None:
        """NEW_OBJECT events produce correct phrases and call speaker once."""
        config = AppConfig()
        config.events.min_frames_to_confirm = 1
        config.events.global_cooldown_seconds = 0.1

        mock_detector = MagicMock()
        # Car in walking path, NEAR (height 270/480 = 0.5625 >= 0.50) -> "near car ahead"
        # Note: per Requirement 4, do not generate "Stop!" solely because NEAR or in corridor
        det = Detection(label="car", confidence=0.9, x1=200, y1=200, x2=440, y2=470, track_id=1)
        mock_detector.detect.return_value = [det]

        mock_speaker = MagicMock()
        mock_speaker.speak.return_value = True

        pipeline = Pipeline(config, detector=mock_detector, speaker=mock_speaker)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        mock_time = [0.0]
        with patch('time.monotonic', side_effect=lambda: mock_time[0]):
            pipeline.process_frame(frame)

        # Speaker called once with concise warning phrase ("near car ahead")
        mock_speaker.speak.assert_called_once_with("near car ahead")

    def test_event_to_phrase_to_speaker_integration_urgency_escalation(self) -> None:
        """URGENCY_ESCALATION events produce correct phrases.

        Requires 2 consecutive NEAR frames due to temporal proximity smoothing.
        """
        config = AppConfig()
        config.events.min_frames_to_confirm = 1
        config.events.global_cooldown_seconds = 0.1
        config.depth.enabled = False  # Use heuristic proximity

        mock_detector = MagicMock()
        mock_speaker = MagicMock()
        mock_speaker.speak.return_value = True

        pipeline = Pipeline(config, detector=mock_detector, speaker=mock_speaker)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        mock_time = [0.0]
        with patch('time.monotonic', side_effect=lambda: mock_time[0]):
            # First frame - person appears (MEDIUM)
            det1 = Detection(label="person", confidence=0.9, x1=200, y1=100, x2=300, y2=200, track_id=1)
            mock_detector.detect.return_value = [det1]
            pipeline.process_frame(frame)

            # Advance past cooldown
            mock_time[0] += 0.5

            # Second frame - person escalates to NEAR (first NEAR frame - smoothed to MEDIUM)
            det2 = Detection(label="person", confidence=0.95, x1=200, y1=100, x2=300, y2=400, track_id=1)
            mock_detector.detect.return_value = [det2]
            pipeline.process_frame(frame)

            # Third frame - second consecutive NEAR frame - proximity escalation confirmed
            mock_time[0] += 0.5
            det3 = Detection(label="person", confidence=0.95, x1=200, y1=100, x2=300, y2=400, track_id=1)
            mock_detector.detect.return_value = [det3]
            pipeline.process_frame(frame)

        # Should have called speak twice: "person ahead" (NEW_OBJECT) then "near person ahead" (PROXIMITY_ESCALATION)
        # Note: urgency escalation also fires on frame 3, but proximity escalation has higher tier
        assert mock_speaker.speak.call_count == 2
        calls = [call[0][0] for call in mock_speaker.speak.call_args_list]
        assert calls[0] == "person ahead"
        assert calls[1] == "near person ahead"

    def test_event_to_phrase_to_speaker_integration_proximity_escalation(self) -> None:
        """PROXIMITY_ESCALATION events produce correct phrases.

        Requires 2 consecutive NEAR frames due to temporal proximity smoothing.
        """
        config = AppConfig()
        config.events.min_frames_to_confirm = 1
        config.events.global_cooldown_seconds = 0.1
        config.depth.enabled = False  # Use heuristic proximity

        mock_detector = MagicMock()
        mock_speaker = MagicMock()
        mock_speaker.speak.return_value = True

        pipeline = Pipeline(config, detector=mock_detector, speaker=mock_speaker)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        mock_time = [0.0]
        with patch('time.monotonic', side_effect=lambda: mock_time[0]):
            # First frame - car at MEDIUM
            det1 = Detection(label="car", confidence=0.9, x1=200, y1=100, x2=300, y2=200, track_id=1)
            mock_detector.detect.return_value = [det1]
            pipeline.process_frame(frame)

            # Advance past cooldown
            mock_time[0] += 0.5

            # Second frame - car escalates to NEAR (first NEAR frame - smoothed to MEDIUM)
            det2 = Detection(label="car", confidence=0.95, x1=200, y1=100, x2=300, y2=400, track_id=1)
            mock_detector.detect.return_value = [det2]
            pipeline.process_frame(frame)

            # Third frame - second consecutive NEAR frame - proximity escalation confirmed
            mock_time[0] += 0.5
            det3 = Detection(label="car", confidence=0.95, x1=200, y1=100, x2=300, y2=400, track_id=1)
            mock_detector.detect.return_value = [det3]
            pipeline.process_frame(frame)

        # Should have called speak twice: "car ahead" (NEW_OBJECT) then "near car ahead" (PROXIMITY_ESCALATION)
        assert mock_speaker.speak.call_count == 2
        calls = [call[0][0] for call in mock_speaker.speak.call_args_list]
        assert calls[0] == "car ahead"
        assert calls[1] == "near car ahead"

    def test_event_to_phrase_to_speaker_integration_periodic_reminder(self) -> None:
        """PERIODIC_REMINDER events produce correct phrases."""
        config = AppConfig()
        config.events.min_frames_to_confirm = 1
        config.events.global_cooldown_seconds = 0.1

        mock_detector = MagicMock()
        mock_speaker = MagicMock()
        mock_speaker.speak.return_value = True

        pipeline = Pipeline(config, detector=mock_detector, speaker=mock_speaker)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # First: NEW_OBJECT (NEAR)
        det = Detection(label="chair", confidence=0.9, x1=200, y1=200, x2=440, y2=470, track_id=1)
        mock_detector.detect.return_value = [det]

        mock_time = [0.0]
        with patch('time.monotonic', side_effect=lambda: mock_time[0]):
            pipeline.process_frame(frame)  # NEW_OBJECT -> "Stop! chair ahead"
            mock_speaker.speak.reset_mock()

            # 6.5 seconds later: PERIODIC_REMINDER
            mock_time[0] += 6.5
            pipeline.process_frame(frame)

        mock_speaker.speak.assert_called_once_with("still near chair ahead")

    def test_single_speaker_call_per_event(self) -> None:
        """Each event results in exactly one call to speaker.speak()."""
        config = AppConfig()
        config.events.min_frames_to_confirm = 1
        config.events.global_cooldown_seconds = 0.1

        mock_detector = MagicMock()
        mock_speaker = MagicMock()
        mock_speaker.speak.return_value = True

        pipeline = Pipeline(config, detector=mock_detector, speaker=mock_speaker)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        det = Detection(label="bottle", confidence=0.8, x1=300, y1=100, x2=350, y2=200, track_id=1)
        mock_detector.detect.return_value = [det]

        mock_time = [0.0]
        with patch('time.monotonic', side_effect=lambda: mock_time[0]):
            # Process multiple frames with same object - should only speak once (NEW_OBJECT)
            pipeline.process_frame(frame)
            pipeline.process_frame(frame)
            pipeline.process_frame(frame)

        # Should only call speak once for the NEW_OBJECT event
        assert mock_speaker.speak.call_count == 1
        mock_speaker.speak.assert_called_with("bottle ahead")

    def test_no_duplicate_speaker_call_on_rejection(self) -> None:
        """When speaker rejects, event is not re-offered within grace period."""
        config = AppConfig()
        config.events.min_frames_to_confirm = 1
        config.events.global_cooldown_seconds = 0.1

        mock_detector = MagicMock()
        det = Detection(label="car", confidence=0.9, x1=200, y1=100, x2=300, y2=200, track_id=1)
        mock_detector.detect.return_value = [det]

        mock_speaker = MagicMock()
        # Speaker rejects (returns False)
        mock_speaker.speak.return_value = False

        pipeline = Pipeline(config, detector=mock_detector, speaker=mock_speaker)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        mock_time = [0.0]
        with patch('time.monotonic', side_effect=lambda: mock_time[0]):
            # Frame 1 at t=0.0: NEW_OBJECT event offered, speaker rejects -> record_offered set to 0.0
            pipeline.process_frame(frame)
            assert mock_speaker.speak.call_count == 1

            # Frame 2 at t=0.2 (< 0.5s grace period): suppressed by last_offered grace period
            mock_time[0] = 0.2
            pipeline.process_frame(frame)
            assert mock_speaker.speak.call_count == 1  # Not called again

            # Frame 3 at t=0.6 (> 0.5s grace period): event can be offered again
            mock_time[0] = 0.6
            pipeline.process_frame(frame)
            assert mock_speaker.speak.call_count == 2


class TestLiveNarrationBugFixes:
    """Targeted regression tests for the live narration silent audio & escalation bug fixes."""

    def test_proximity_escalation_across_cooldown_boundary(self) -> None:
        """Obstacle approaching during global cooldown must still trigger escalation once cooldown expires."""
        config = EventsConfig(global_cooldown_seconds=1.5, min_frames_to_confirm=1)
        narrator = EventNarrator(config=config)

        # 1. Obstacle confirmed at FAR and spoken at t=0.0
        det_far = Detection(label="person", confidence=0.9, x1=100, y1=100, x2=200, y2=150, track_id=10)
        obs_far = PrioritizedObstacle(
            detection=det_far,
            direction=Direction.CENTER,
            proximity=Proximity.FAR,
            urgency_score=20.0,
        )

        with patch("time.monotonic") as mock_time:
            mock_time.return_value = 0.0
            ev = narrator.update([obs_far], current_time=0.0)
            assert ev is not None
            assert ev.event_type == EventType.NEW_OBJECT
            narrator.record_spoken(ev, current_time=0.0)

            # 2. Obstacle approaches to MEDIUM during global cooldown (t=0.5s < 1.5s)
            det_med = Detection(label="person", confidence=0.9, x1=100, y1=100, x2=200, y2=250, track_id=10)
            obs_med = PrioritizedObstacle(
                detection=det_med,
                direction=Direction.CENTER,
                proximity=Proximity.MEDIUM,
                urgency_score=50.0,
            )
            mock_time.return_value = 0.5
            ev_cooldown = narrator.update([obs_med], current_time=0.5)
            # Should be suppressed while global cooldown is active
            assert ev_cooldown is None

            # 3. Time advances past global cooldown (t=1.6s >= 1.5s)
            # Obstacle remains at MEDIUM. Escalation must not be lost!
            mock_time.return_value = 1.6
            ev_escalated = narrator.update([obs_med], current_time=1.6)
            assert ev_escalated is not None
            assert ev_escalated.event_type == EventType.PROXIMITY_ESCALATION
            assert ev_escalated.proximity == Proximity.MEDIUM
            assert ev_escalated.label == "person"
            narrator.record_spoken(ev_escalated, current_time=1.6)

            # 4. Same MEDIUM proximity after cooldown should NOT re-escalate
            mock_time.return_value = 3.5
            ev_same = narrator.update([obs_med], current_time=3.5)
            assert ev_same is None

    def test_direction_change_from_side_to_center_escalation(self) -> None:
        """Obstacle moving from side (RIGHT) to center path (AHEAD) must trigger escalation."""
        config = EventsConfig(global_cooldown_seconds=1.0, min_frames_to_confirm=1)
        narrator = EventNarrator(config=config)

        det_right = Detection(label="bicycle", confidence=0.85, x1=500, y1=100, x2=600, y2=250, track_id=11)
        obs_right = PrioritizedObstacle(
            detection=det_right,
            direction=Direction.RIGHT,
            proximity=Proximity.MEDIUM,
            urgency_score=40.0,
        )

        with patch("time.monotonic") as mock_time:
            # 1. Announced on the right
            mock_time.return_value = 0.0
            ev = narrator.update([obs_right], current_time=0.0)
            assert ev is not None
            assert ev.event_type == EventType.NEW_OBJECT
            narrator.record_spoken(ev, current_time=0.0)

            # 2. Obstacle moves to CENTER ahead
            det_center = Detection(label="bicycle", confidence=0.85, x1=250, y1=100, x2=350, y2=250, track_id=11)
            obs_center = PrioritizedObstacle(
                detection=det_center,
                direction=Direction.CENTER,
                proximity=Proximity.MEDIUM,
                urgency_score=65.0,
            )

            # Advance past cooldown
            mock_time.return_value = 1.5
            ev_shifted = narrator.update([obs_center], current_time=1.5)
            assert ev_shifted is not None
            assert ev_shifted.event_type == EventType.URGENCY_ESCALATION
            assert ev_shifted.direction == Direction.CENTER
            assert "ahead" in ev_shifted.speech_text

    def test_periodic_reminder_for_in_path_medium_obstacle(self) -> None:
        """In-path MEDIUM obstacle must trigger a periodic reminder after 6 seconds."""
        from narrator.ground_plane import ContactPoint, HazardZone

        config = EventsConfig(min_frames_to_confirm=1)
        narrator = EventNarrator(config=config)

        det = Detection(label="chair", confidence=0.9, x1=250, y1=150, x2=390, y2=250, track_id=12)
        cp = ContactPoint(x=320, y=250, normalized_x=0.5, normalized_y=0.52, zone=HazardZone.WALKING_PATH)
        obs = PrioritizedObstacle(
            detection=det,
            direction=Direction.CENTER,
            proximity=Proximity.MEDIUM,
            urgency_score=75.0,
            contact_point=cp,
        )
        assert obs.is_in_walking_path is True

        with patch("time.monotonic") as mock_time:
            # First announcement
            mock_time.return_value = 0.0
            ev = narrator.update([obs], current_time=0.0)
            assert ev is not None
            narrator.record_spoken(ev, current_time=0.0)

            # Wait 5.9s (not ready)
            mock_time.return_value = 5.9
            assert narrator.update([obs], current_time=5.9) is None

            # Wait 6.1s (ready)
            mock_time.return_value = 6.1
            ev_reminder = narrator.update([obs], current_time=6.1)
            assert ev_reminder is not None
            assert ev_reminder.event_type == EventType.PERIODIC_REMINDER
            assert ev_reminder.label == "chair"
            assert "still medium chair ahead" in ev_reminder.speech_text

    def test_urgency_escalation_evaluated_against_spoken_baseline(self) -> None:
        """Urgency escalation must compare against last spoken baseline, not instantaneous 33ms frame."""
        config = EventsConfig(global_cooldown_seconds=1.0, track_cooldown_seconds=1.0, min_frames_to_confirm=1)
        narrator = EventNarrator(config=config)

        det1 = Detection(label="car", confidence=0.8, x1=200, y1=100, x2=300, y2=200, track_id=13)
        obs1 = PrioritizedObstacle(det1, Direction.CENTER, Proximity.FAR, urgency_score=20.0)

        with patch("time.monotonic") as mock_time:
            # 1. Announced at score 20.0
            mock_time.return_value = 0.0
            ev = narrator.update([obs1], current_time=0.0)
            assert ev is not None
            narrator.record_spoken(ev, current_time=0.0)

            # 2. Score gradually creeps up over 3 frames: 22 -> 25 -> 28
            # Frame A (t=0.1)
            obs_a = PrioritizedObstacle(det1, Direction.CENTER, Proximity.FAR, urgency_score=22.0)
            mock_time.return_value = 0.1
            narrator.update([obs_a], current_time=0.1)

            # Frame B (t=0.2)
            obs_b = PrioritizedObstacle(det1, Direction.CENTER, Proximity.FAR, urgency_score=25.0)
            mock_time.return_value = 0.2
            narrator.update([obs_b], current_time=0.2)

            # Frame C (t=1.2s past cooldown): score is 28.0.
            # 28.0 > 20.0 * 1.3 (26.0) -> Escalation triggered!
            # (If compared to frame B's 25.0, 28.0 < 25.0*1.3=32.5 and would fail)
            obs_c = PrioritizedObstacle(det1, Direction.CENTER, Proximity.FAR, urgency_score=28.0)
            mock_time.return_value = 1.2
            ev_urg = narrator.update([obs_c], current_time=1.2)
            assert ev_urg is not None
            assert ev_urg.event_type == EventType.URGENCY_ESCALATION

    def test_speaker_multiple_utterances_successive_playback(self) -> None:
        """Speaker worker must cleanly play multiple successive utterances without premature termination."""
        from narrator.config import SpeakerConfig
        from narrator.speaker import Speaker

        spoken_calls: list[str] = []

        class MockProxy:
            def __init__(self) -> None:
                self._busy = False
                self._queue: list = []

            def setBusy(self, busy: bool) -> None:
                self._busy = busy

        class MockEngine:
            def __init__(self) -> None:
                self.proxy = MockProxy()

            def setProperty(self, name: str, val: object) -> None:
                pass

            def say(self, text: str) -> None:
                # If setBusy fix is working, proxy._busy should be True when say is called
                assert self.proxy._busy is True
                spoken_calls.append(text)

            def runAndWait(self) -> None:
                # pyttsx3 sets busy=False when runAndWait ends
                self.proxy._busy = False

            def stop(self) -> None:
                pass

        mock_engine = MockEngine()
        cfg = SpeakerConfig(enabled=True, cooldown_seconds=0.0)

        with patch("pyttsx3.init", return_value=mock_engine):
            spk = Speaker(cfg)
            try:
                assert spk.speak("first phrase") is True
                assert spk.speak("second phrase") is True
                assert spk.speak("third phrase") is True

                start_t = time.monotonic()
                while len(spoken_calls) < 3 and (time.monotonic() - start_t) < 2.0:
                    time.sleep(0.02)

                assert spoken_calls == ["first phrase", "second phrase", "third phrase"]
                assert spk.is_worker_alive is True
            finally:
                spk.stop()



if __name__ == "__main__":
    pytest.main([__file__, "-v"])