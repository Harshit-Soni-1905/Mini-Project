#!/usr/bin/env python3
"""Simple test to verify our fixes work."""

import sys
from unittest.mock import patch

# Add project to path
sys.path.insert(0, '.')

from src.narrator.events import EventNarrator, EventsConfig
from src.narrator.detector import Detection
from src.narrator.prioritizer import PrioritizedObstacle
from src.narrator.direction import Direction
from src.narrator.distance import Proximity
from src.narrator.events import EventType

def test_global_cooldown():
    """Test global cooldown enforcement."""
    print("Testing global cooldown...")
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
        # Frame 1
        narrator.update([obstacle1], current_time=0.0)
        # Frame 2 - should trigger NEW_OBJECT
        events = narrator.update([obstacle1], current_time=0.1)
        assert events is not None, "Should have NEW_OBJECT event at frame 2"
        assert events.event_type == EventType.NEW_OBJECT

        # Frame 3 (0.2s) - should be blocked by global cooldown
        mock_time.return_value = 0.2
        events = narrator.update([obstacle2], current_time=0.2)
        assert events is None, f"Should be blocked by global cooldown, got {events}"

        # Frame 4 (2.2s) - should be allowed after global cooldown
        mock_time.return_value = 2.2
        events = narrator.update([obstacle2], current_time=2.2)
        assert events is not None, "Should be allowed after global cooldown"
        assert events.label == "car"

    print("✓ Global cooldown test PASSED")

def test_track_reappearance():
    """Test track reappearing after grace period."""
    print("Testing track reappearance...")
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
        # Frame 1
        narrator.update([obstacle], current_time=0.0)
        # Frame 2 - should trigger NEW_OBJECT
        events = narrator.update([obstacle], current_time=0.1)
        assert events is not None and events.event_type == EventType.NEW_OBJECT

        # Remove track
        mock_time.return_value = 0.2
        narrator.update([], current_time=0.2)

        # Wait beyond grace period
        mock_time.return_value = 2.0
        narrator.update([], current_time=2.0)

        # Reintroduce track - frame 1 of return
        mock_time.return_value = 2.1
        narrator.update([obstacle], current_time=2.1)
        events = narrator.update([obstacle], current_time=2.1)
        assert events is None, "First frame of reappearance should not trigger event"

        # Frame 2 of return - should trigger NEW_OBJECT again
        mock_time.return_value = 2.2
        events = narrator.update([obstacle], current_time=2.2)
        assert events is not None, "Second frame of reappearance should trigger NEW_OBJECT"
        assert events.event_type == EventType.NEW_OBJECT

    print("✓ Track reappearance test PASSED")

def test_should_announce():
    """Test _should_announce logic."""
    print("Testing _should_announce...")
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
        # First announcement should be allowed
        assert narrator._should_announce(event, 0.0) is True
        narrator._record_spoken(event, 0.0)

        # Immediately after - should be blocked by global cooldown
        mock_time.return_value = 0.5
        assert narrator._should_announce(event, 0.5) is False

        # After global cooldown expires - should be allowed
        mock_time.return_value = 1.1
        assert narrator._should_announce(event, 1.1) is True
        narrator._record_spoken(event, 1.1)

        # Same track/proximity immediately after - should be blocked by per-track cooldown
        mock_time.return_value = 1.2
        assert narrator._should_announce(event, 1.2) is False

        # Different proximity should work (if proximity changed)
        det_med = Detection(label="car", confidence=0.9, x1=100, y1=150, x2=200, y2=250, track_id=55)
        obstacle_med = PrioritizedObstacle(
            detection=det_med,
            direction=Direction.CENTER,
            proximity=Proximity.MEDIUM,
            urgency_score=70.0,
        )
        event_med = narrator._make_event(obstacle_med, EventType.PROXIMITY_ESCALATION)

        # Advance time past global cooldown before testing different proximity
        mock_time.return_value = 2.2  # Last global speech was at 1.1, need >= 2.1
        assert narrator._should_announce(event_med, 2.2) is True

        # Wait for per-track cooldown to expire
        mock_time.return_value = 3.2
        # Same proximity again - should now be allowed
        assert narrator._should_announce(event, 3.2) is True

    print("✓ _should_announce test PASSED")

def test_urgency_escalation():
    """Test urgency escalation event."""
    print("Testing urgency escalation...")
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
        test_global_cooldown()
        test_track_reappearance()
        test_should_announce()
        test_urgency_escalation()
        print("\n🎉 All tests PASSED! The fixes are working correctly.")
    except Exception as e:
        print(f"\n❌ Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)