#!/usr/bin/env python3
"""Run specific tests to verify fixes."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from unittest.mock import patch
from src.narrator.events import EventNarrator, EventsConfig
from src.narrator.detector import Detection
from src.narrator.prioritizer import PrioritizedObstacle
from src.narrator.direction import Direction
from src.narrator.distance import Proximity
from src.narrator.events import EventType
from src.narrator.metrics import PerformanceMonitor

def test_global_cooldown_enforcement():
    """Test that global cooldown prevents speech within threshold."""
    print("Testing global_cooldown_enforcement...")
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
        # First obstacle should trigger event (establish track and confirm)
        narrator.update([obstacle1], current_time=0.0)  # Frame 1
        events = narrator.update([obstacle1], current_time=0.1)  # Frame 2 - should trigger NEW_OBJECT
        assert events is not None, "First NEW_OBJECT event should not be None"
        first_event_time = 0.1

        # Immediately try second obstacle - should be blocked by global cooldown (2.0s in this config)
        mock_time.return_value = 0.2
        events = narrator.update([obstacle2], current_time=0.2)
        assert events is None, f"Second obstacle should be blocked by global cooldown, got {events}"

        # Wait until global cooldown expires (global cooldown is 2.0s)
        mock_time.return_value = 2.2  # 2.2 - 0.1 = 2.1 >= 2.0
        events = narrator.update([obstacle2], current_time=2.2)
        assert events is not None, "Second obstacle should be allowed after global cooldown expires"
        assert events.label == "car", f"Expected car, got {events.label}"
        print("✓ Global cooldown enforcement test PASSED")

def test_track_reappearance_after_grace_period_is_new_object():
    """Test that track reappearing after grace period is treated as new."""
    print("Testing track_reappearance_after_grace_period_is_new_object...")
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
        assert events is not None and events.event_type == EventType.NEW_OBJECT, "First NEW_OBJECT event should not be None"

        # Now remove track
        mock_time.return_value = 0.2
        narrator.update([], current_time=0.2)  # Track disappears

        # Wait beyond grace period
        mock_time.return_value = 2.0  # 1.8 seconds later
        narrator.update([], current_time=2.0)  # Track purged from memory

        # Reintroduce track - should be treated as new object again
        mock_time.return_value = 2.1
        # First frame of reappearance
        narrator.update([obstacle], current_time=2.1)  # frame 1 of return
        events = narrator.update([obstacle], current_time=2.1)
        assert events is None, "First frame of reappearance should not trigger event (not confirmed yet)"

        # Second frame - should trigger NEW_OBJECT again
        mock_time.return_value = 2.2
        events = narrator.update([obstacle], current_time=2.2)
        assert events is not None, "Second frame of reappearance should trigger NEW_OBJECT"
        assert events.event_type == EventType.NEW_OBJECT, f"Expected NEW_OBJECT, got {events.event_type}"
        print("✓ Track reappearance test PASSED")

def test_should_announce_respects_cooldowns():
    """Test that _should_announce properly enforces global and per-track cooldowns."""
    print("Testing test_should_announce_respects_cooldowns...")
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
        assert narrator._should_announce(event, 0.0) is True, "First announcement should be allowed"
        # Record that we spoke
        narrator._record_spoken(event, 0.0)

        # Immediately after - should be blocked by global cooldown (1.0s)
        mock_time.return_value = 0.5
        assert narrator._should_announce(event, 0.5) is False, "Should be blocked by global cooldown"

        # After global cooldown expires - should be allowed
        mock_time.return_value = 1.1
        assert narrator._should_announce(event, 1.1) is True, "Should be allowed after global cooldown"
        narrator._record_spoken(event, 1.1)

        # Now test per-track cooldown
        # Same track/proximity immediately after - should be blocked by per-track cooldown (2.0s)
        mock_time.return_value = 1.2
        assert narrator._should_announce(event, 1.2) is False, "Should be blocked by per-track cooldown"

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
        result = narrator._should_announce(event_med, 2.2)
        assert result is True, f"Different proximity should be allowed after global cooldown, got {result}"

        # Wait for per-track cooldown to expire
        mock_time.return_value = 3.2
        # Same proximity again - should now be allowed
        result = narrator._should_announce(event, 3.2)
        assert result is True, f"Same proximity should be allowed after per-track cooldown expires, got {result}"
        print("✓ Should announce respects cooldowns test PASSED")

def test_get_summary_returns_correct_dictionary():
    """Test that get_summary returns correct dictionary values."""
    print("Testing test_get_summary_returns_correct_dictionary...")
    monitor = PerformanceMonitor(window_size=5, warmup_frames=2)  # Using warmup_frames=2 as in original test
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
        print(f"Summary: {summary}")

        # With warmup_frames=2, valid latencies start from index 1 (exclude first frame)
        # valid = list(self._inference_latencies)[self._warmup_frames - 1 :] = [10, 20, 30, 40][1:] = [20, 30, 40]
        # avg = (20+30+40)/3 = 30.0
        # avg_total = (60+70+80)/3 = 70.0
        expected_fps = 10.0  # 1/(0.1) = 10.0
        expected_avg_inference = 30.0  # (20+30+40)/3
        expected_avg_total = 70.0  # (60+70+80)/3
        expected_active_tracks = 2

        assert summary["fps"] == expected_fps, f"Expected FPS {expected_fps}, got {summary['fps']}"
        assert summary["avg_inference_ms"] == expected_avg_inference, f"Expected avg_inference_ms {expected_avg_inference}, got {summary['avg_inference_ms']}"
        assert summary["avg_total_ms"] == expected_avg_total, f"Expected avg_total_ms {expected_avg_total}, got {summary['avg_total_ms']}"
        assert summary["active_tracks"] == expected_active_tracks, f"Expected active_tracks {expected_active_tracks}, got {summary['active_tracks']}"
        assert summary["device"] == "CUDA:1", f"Expected device CUDA:1, got {summary['device']}"
        assert summary["warmed_up"] is True, "Should be warmed up after 4 frames with warmup_frames=2"
        print("✓ Get summary test PASSED")

def test_urgency_escalation_event():
    """Test that urgency escalation event is triggered correctly."""
    print("Testing test_urgency_escalation_event...")
    config = EventsConfig()
    narrator = EventNarrator(config=config)

    det = Detection(label="chair", confidence=0.8, x1=50, y1=100, x2=150, y2=200, track_id=21)
    obstacle_low = PrioritizedObstacle(
        detection=det,
        direction=Direction.LEFT,
        proximity=Proximity.FAR,
        urgency_score=20.0,
    )

    with patch('time.monotonic') as mock_time:
        mock_time.return_value = 0.0
        # Establish track for 2 frames to confirm
        narrator.update([obstacle_low], current_time=0.0)  # Frame 1
        narrator.update([obstacle_low], current_time=0.1)  # Frame 2 - should trigger NEW_OBJECT

        # Now trigger urgency escalation after BOTH global cooldown (1.5s) AND track cooldown (4.0s)
        # Need to wait until now - last_spoken >= track_cooldown (4.0s)
        det_high = Detection(label="chair", confidence=0.95, x1=50, y1=100, x2=150, y2=200, track_id=21)  # SAME track_id
        obstacle_high = PrioritizedObstacle(
            detection=det_high,
            direction=Direction.LEFT,
            proximity=Proximity.FAR,
            urgency_score=50.0,  # > 1.3 * 20.0 = 26.0
        )
        mock_time.return_value = 5.0  # 5.0 - 0.1 = 4.9 > 4.0 (track cooldown)
        events = narrator.update([obstacle_high], current_time=5.0)
        assert events is not None, "Urgency escalation event should not be None"
        assert events.event_type == EventType.URGENCY_ESCALATION, f"Expected URGENCY_ESCALATION, got {events.event_type}"
        assert events.label == "chair", f"Expected chair, got {events.label}"
        print("✓ Urgency escalation test PASSED")

if __name__ == "__main__":
    try:
        test_global_cooldown_enforcement()
        test_track_reappearance_after_grace_period_is_new_object()
        test_should_announce_respects_cooldowns()
        test_get_summary_returns_correct_dictionary()
        test_urgency_escalation_event()
        print("\n🎉 All tests PASSED!")
    except Exception as e:
        print(f"\n❌ Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)