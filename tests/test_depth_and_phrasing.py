"""Unit tests for Phase 4 (Depth Estimation) & Phase 5 (Spoken Warning Policy & Preemption).

Covers:
- GeometricDepthEstimator depth maps and perspective calculation
- Relative depth sampling and DistanceCategory classification (Immediate, Near, Medium, Far)
- Spoken warning phrase formatting across urgency tiers (Urgent, Warning, Advisory)
- Speaker queue clearing, audio preemption, and urgent speech dispatch
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from narrator.config import SpeakerConfig
from narrator.depth import DepthCategory, GeometricDepthEstimator
from narrator.detector import Detection
from narrator.direction import Direction
from narrator.distance import Proximity
from narrator.events import EventType
from narrator.ground_plane import ContactPoint, HazardZone
from narrator.phrasing import AlertLevel, format_spoken_alert
from narrator.prioritizer import PrioritizedObstacle
from narrator.speaker import Speaker


class TestDepthEstimation:
    def test_geometric_depth_map_dimensions(self) -> None:
        estimator = GeometricDepthEstimator(horizon_y=0.40)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        depth_map = estimator.estimate_depth(frame)

        assert depth_map.shape == (480, 640)
        assert depth_map.dtype == np.float32
        assert estimator.is_inverse_depth is True

        # Closeness at bottom should be maximum (1.0), and 0.0 at/above horizon
        assert depth_map[479, 320] == pytest.approx(1.0, abs=1e-3)
        assert depth_map[0, 320] == 0.0
        assert depth_map[int(480 * 0.40), 320] == pytest.approx(0.0, abs=0.01)

    def test_sample_obstacle_depth_and_classification(self) -> None:
        estimator = GeometricDepthEstimator(horizon_y=0.30)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        depth_map = estimator.estimate_depth(frame)

        # Obstacle near bottom of screen (feet)
        det_feet = Detection("chair", 0.9, x1=200, y1=380, x2=400, y2=470)
        depth_feet = estimator.get_obstacle_depth(depth_map, det_feet)
        cat_feet = estimator.classify_distance(depth_feet)
        assert cat_feet in (DepthCategory.IMMEDIATE, DepthCategory.NEAR)

        # Obstacle far away near horizon
        det_far = Detection("chair", 0.8, x1=200, y1=150, x2=250, y2=190)
        depth_far = estimator.get_obstacle_depth(depth_map, det_far)
        cat_far = estimator.classify_distance(depth_far)
        assert cat_far == DepthCategory.FAR


class TestWarningPhrasing:
    def test_urgent_phrase_in_walking_path(self) -> None:
        det = Detection("chair", 0.9, 200, 200, 440, 470)
        cp = ContactPoint(x=320, y=470, normalized_x=0.5, normalized_y=0.98, zone=HazardZone.WALKING_PATH)
        obs = PrioritizedObstacle(
            detection=det,
            direction=Direction.CENTER,
            proximity=Proximity.NEAR,
            urgency_score=150.0,
            contact_point=cp,
        )

        # Urgent auto-inferred because NEAR + WALKING_PATH
        phrase = format_spoken_alert(obs)
        assert phrase == "Stop! chair ahead"

    def test_warning_phrase(self) -> None:
        det = Detection("car", 0.9, 100, 100, 200, 200)
        obs = PrioritizedObstacle(
            detection=det,
            direction=Direction.LEFT,
            proximity=Proximity.NEAR,
            urgency_score=110.0,
        )
        phrase = format_spoken_alert(obs, alert_level=AlertLevel.WARNING)
        assert phrase == "near car left"

    def test_advisory_phrase(self) -> None:
        det = Detection("person", 0.8, 500, 200, 580, 300)
        obs = PrioritizedObstacle(
            detection=det,
            direction=Direction.RIGHT,
            proximity=Proximity.FAR,
            urgency_score=20.0,
        )
        phrase = format_spoken_alert(obs)
        assert phrase == "person right"

    def test_periodic_reminder_phrase(self) -> None:
        det = Detection("chair", 0.9, 200, 200, 440, 470)
        obs = PrioritizedObstacle(
            detection=det,
            direction=Direction.CENTER,
            proximity=Proximity.NEAR,
            urgency_score=100.0,
        )
        phrase = format_spoken_alert(obs, event_type=EventType.PERIODIC_REMINDER)
        assert phrase == "still near chair ahead"

    def test_new_object_near_phrase(self) -> None:
        """NEW_OBJECT event with NEAR proximity should use 'near' prefix (when not in walking path)."""
        det = Detection("car", 0.9, 100, 300, 200, 450)
        cp = ContactPoint(x=150, y=450, normalized_x=0.23, normalized_y=0.94, zone=HazardZone.PERIPHERAL)
        obs = PrioritizedObstacle(
            detection=det,
            direction=Direction.LEFT,
            proximity=Proximity.NEAR,
            urgency_score=120.0,
            contact_point=cp,
        )
        phrase = format_spoken_alert(obs, event_type=EventType.NEW_OBJECT)
        assert phrase == "near car left"

    def test_new_object_medium_phrase(self) -> None:
        """NEW_OBJECT event with MEDIUM proximity should not use 'near' prefix."""
        det = Detection("person", 0.8, 300, 100, 400, 200)
        obs = PrioritizedObstacle(
            detection=det,
            direction=Direction.CENTER,
            proximity=Proximity.MEDIUM,
            urgency_score=60.0,
        )
        phrase = format_spoken_alert(obs, event_type=EventType.NEW_OBJECT)
        assert phrase == "person ahead"

    def test_new_object_far_phrase(self) -> None:
        """NEW_OBJECT event with FAR proximity should not use 'near' prefix."""
        det = Detection("bicycle", 0.7, 500, 50, 580, 150)
        obs = PrioritizedObstacle(
            detection=det,
            direction=Direction.RIGHT,
            proximity=Proximity.FAR,
            urgency_score=30.0,
        )
        phrase = format_spoken_alert(obs, event_type=EventType.NEW_OBJECT)
        assert phrase == "bicycle right"

    def test_left_direction_phrasing(self) -> None:
        """LEFT direction should produce 'left' in output."""
        det = Detection("chair", 0.9, 50, 200, 150, 300)
        obs = PrioritizedObstacle(
            detection=det,
            direction=Direction.LEFT,
            proximity=Proximity.NEAR,
            urgency_score=100.0,
        )
        phrase = format_spoken_alert(obs)
        assert "left" in phrase
        assert phrase == "near chair left"

    def test_ahead_direction_phrasing(self) -> None:
        """CENTER (ahead) direction should produce 'ahead' in output."""
        det = Detection("chair", 0.9, 250, 200, 390, 300)
        obs = PrioritizedObstacle(
            detection=det,
            direction=Direction.CENTER,
            proximity=Proximity.NEAR,
            urgency_score=100.0,
        )
        phrase = format_spoken_alert(obs)
        assert "ahead" in phrase
        assert phrase == "near chair ahead"

    def test_right_direction_phrasing(self) -> None:
        """RIGHT direction should produce 'right' in output."""
        det = Detection("chair", 0.9, 490, 200, 590, 300)
        obs = PrioritizedObstacle(
            detection=det,
            direction=Direction.RIGHT,
            proximity=Proximity.NEAR,
            urgency_score=100.0,
        )
        phrase = format_spoken_alert(obs)
        assert "right" in phrase
        assert phrase == "near chair right"

    def test_obstacle_in_walking_path_urgent(self) -> None:
        """Obstacle in walking path with NEAR proximity triggers URGENT alert."""
        det = Detection("chair", 0.9, 200, 300, 440, 470)
        cp = ContactPoint(x=320, y=470, normalized_x=0.5, normalized_y=0.98, zone=HazardZone.WALKING_PATH)
        obs = PrioritizedObstacle(
            detection=det,
            direction=Direction.CENTER,
            proximity=Proximity.NEAR,
            urgency_score=150.0,
            contact_point=cp,
        )
        phrase = format_spoken_alert(obs)
        assert phrase == "Stop! chair ahead"

    def test_obstacle_in_walking_path_medium_not_urgent(self) -> None:
        """Obstacle in walking path with MEDIUM proximity triggers WARNING (not URGENT)."""
        det = Detection("chair", 0.9, 200, 150, 440, 250)
        cp = ContactPoint(x=320, y=250, normalized_x=0.5, normalized_y=0.52, zone=HazardZone.WALKING_PATH)
        obs = PrioritizedObstacle(
            detection=det,
            direction=Direction.CENTER,
            proximity=Proximity.MEDIUM,
            urgency_score=80.0,
            contact_point=cp,
        )
        # MEDIUM + walking path -> WARNING level -> "chair ahead"
        phrase = format_spoken_alert(obs)
        assert phrase == "chair ahead"

    def test_obstacle_outside_walking_path(self) -> None:
        """Obstacle outside walking path (PERIPHERAL) with NEAR proximity triggers WARNING."""
        det = Detection("chair", 0.9, 50, 300, 150, 470)
        cp = ContactPoint(x=100, y=470, normalized_x=0.16, normalized_y=0.98, zone=HazardZone.PERIPHERAL)
        obs = PrioritizedObstacle(
            detection=det,
            direction=Direction.LEFT,
            proximity=Proximity.NEAR,
            urgency_score=90.0,
            contact_point=cp,
        )
        # NEAR but peripheral -> WARNING -> "near chair left"
        phrase = format_spoken_alert(obs)
        assert phrase == "near chair left"

    def test_obstacle_elevated_zone(self) -> None:
        """Obstacle in ELEVATED zone (tabletop) with MEDIUM proximity triggers ADVISORY."""
        det = Detection("bottle", 0.9, 250, 50, 350, 120)
        cp = ContactPoint(x=300, y=120, normalized_x=0.47, normalized_y=0.25, zone=HazardZone.ELEVATED)
        obs = PrioritizedObstacle(
            detection=det,
            direction=Direction.CENTER,
            proximity=Proximity.MEDIUM,
            urgency_score=30.0,
            contact_point=cp,
        )
        # ELEVATED -> ADVISORY -> "bottle ahead"
        phrase = format_spoken_alert(obs)
        assert phrase == "bottle ahead"

    def test_missing_contact_point(self) -> None:
        """Missing contact_point (None) should default to ADVISORY level."""
        det = Detection("cup", 0.8, 300, 100, 350, 150)
        obs = PrioritizedObstacle(
            detection=det,
            direction=Direction.CENTER,
            proximity=Proximity.FAR,
            urgency_score=20.0,
            contact_point=None,
        )
        phrase = format_spoken_alert(obs)
        assert phrase == "cup ahead"

    def test_urgency_escalation_uses_warning_level(self) -> None:
        """URGENCY_ESCALATION event with NEAR proximity uses WARNING phrasing."""
        det = Detection("car", 0.9, 200, 100, 400, 300)
        obs = PrioritizedObstacle(
            detection=det,
            direction=Direction.CENTER,
            proximity=Proximity.NEAR,
            urgency_score=110.0,
        )
        phrase = format_spoken_alert(obs, event_type=EventType.URGENCY_ESCALATION)
        # NEAR proximity -> "near car ahead"
        assert phrase == "near car ahead"

    def test_proximity_escalation_uses_warning_level(self) -> None:
        """PROXIMITY_ESCALATION event with NEAR proximity uses WARNING phrasing."""
        det = Detection("person", 0.8, 100, 200, 200, 350)
        obs = PrioritizedObstacle(
            detection=det,
            direction=Direction.LEFT,
            proximity=Proximity.NEAR,
            urgency_score=100.0,
        )
        phrase = format_spoken_alert(obs, event_type=EventType.PROXIMITY_ESCALATION)
        assert phrase == "near person left"


class TestSpeakerPreemption:
    def test_clear_queue(self) -> None:
        cfg = SpeakerConfig(enabled=True, cooldown_seconds=0.0)
        with patch.object(Speaker, "_start_worker"):
            spk = Speaker(cfg)
            try:
                # Fill queue
                spk._queue.put("msg1")
                spk._queue.put("msg2")
                dropped = spk.clear_queue()
                assert dropped >= 2
                assert spk.queue_size == 0
            finally:
                spk.stop()

    def test_urgent_speak_preempts_queue(self) -> None:
        cfg = SpeakerConfig(enabled=True, cooldown_seconds=10.0)
        with patch.object(Speaker, "_start_worker"):
            spk = Speaker(cfg)
            try:
                # Send normal message
                assert spk.speak("first warning") is True

                # Repeated normal message suppressed by cooldown
                assert spk.speak("first warning") is False

                # Urgent message bypasses duplicate cooldown and enqueues immediately
                assert spk.speak("first warning", urgent=True) is True
            finally:
                spk.stop()
