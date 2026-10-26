#!/usr/bin/env python3
"""Simple verification script to test the fixed logic."""

import sys
from unittest.mock import patch

# Add the project root to the path
sys.path.insert(0, 'C:\\MyWork\\Mini Project')

from src.narrator.events import EventNarrator, EventsConfig
from src.narrator.detector import Detection
from src.narrator.prioritizer import PrioritizedObstacle
from src.narrator.direction import Direction
from src.narrator.distance import Proximity
from src.narrator.events import EventType

def test_global_cooldown_logic():
    """Test the specific logic that was failing."""
    print("Testing global cooldown enforcement logic...")

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
        result1 = narrator.update([obstacle1], current_time=0.0)  # Frame 1
        print(f"Frame 1 result: {result1}")

        events = narrator.update([obstacle1], current_time=0.1)  # Frame 2 - should trigger NEW_OBJECT
        print(f"Frame 2 result: {events}")
        assert events is not None, "Frame 2 should return NEW_OBJECT event"
        first_event_time = 0.1

        # Immediately try second obstacle - should be blocked by global cooldown (2.0s in this config)
        mock_time.return_value = 0.2
        events = narrator.update([obstacle2], current_time=0.2)
        print(f"Frame 3 (0.2s) result: {events}")
        assert events is None, f"Should be blocked by global cooldown, got {events}"

        # Wait until global cooldown expires (global cooldown is 2.0s)
        mock_time.return_value = 2.2  # 2.2 - 0.1 = 2.1 >= 2.0
        events = narrator.update([obstacle2], current_time=2.2)
        print(f"Frame 4 (2.2s) result: {events}")
        assert events is not None, "Should be allowed after global cooldown expires"
        assert events.label == "car", f"Expected car, got {events.label}"

        print("✓ Global cooldown enforcement logic works correctly!")

def test_track_reappearance_logic():
    """Test track reappearance logic."""
    print("\nTesting track reappearance logic...")

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
        result1 = narrator.update([obstacle], current_time=0.0)  # frame 1
        print(f"Frame 1 result: {result1}")

        events = narrator.update([obstacle], current_time=0.1)  # frame 2 - should trigger NEW_OBJECT
        print(f"Frame 2 result: {events}")
        assert events is not None and events.event_type == EventType.NEW_OBJECT, "Frame 2 should trigger NEW_OBJECT"

        # Now remove track
        mock_time.return_value = 0.2
        result3 = narrator.update([], current_time=0.2)  # Track disappears
        print(f"Frame 3 result (disappeared): {result3}")

        # Wait beyond grace period
        mock_time.return_value = 2.0  # 1.8 seconds later
        result4 = narrator.update([], current_time=2.0)  # Track purged from memory
        print(f"Frame 4 result (after grace): {result4}")

        # Reintroduce track - should be treated as new object again
        mock_time.return_value = 2.1
        # First frame of reappearance
        result5 = narrator.update([obstacle], current_time=2.1)  # frame 1 of return
        print(f"Frame 5 result (reappearance frame 1): {result5}")
        assert result5 is None, "First frame of reappearance should not trigger event (not confirmed yet)"

        # Second frame - should trigger NEW_OBJECT again
        mock_time.return_value = 2.2
        events = narrator.update([obstacle], current_time=2.2)
        print(f"Frame 6 result (reappearance frame 2): {events}")
        assert events is not None, "Second frame of reappearance should trigger NEW_OBJECT"
        assert events.event_type == EventType.NEW_OBJECT, f"Expected NEW_OBJECT, got {events.event_type}"

        print("✓ Track reappearance logic works correctly!")

def test_should_announce_logic():
    """Test _should_announce logic."""
    print("\nTesting _should_announce logic...")

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
        result1 = narrator._should_announce(event, 0.0)
        print(f"Time 0.0 - should announce: {result1}")
        assert result1 is True, "First announcement should be allowed"
        # Record that we spoke
        narrator._record_spoken(event, 0.0)

        # Immediately after - should be blocked by global cooldown (1.0s)
        mock_time.return_value = 0.5
        result2 = narrator._should_announce(event, 0.5)
        print(f"Time 0.5 - should announce: {result2}")
        assert result2 is False, "Should be blocked by global cooldown"

        # After global cooldown expires - should be allowed
        mock_time.return_value = 1.1
        result3 = narrator._should_announce(event, 1.1)
        print(f"Time 1.1 - should announce: {result3}")
        assert result3 is True, "Should be allowed after global cooldown"
        narrator._record_spoken(event, 1.1)

        # Now test per-track cooldown
        # Same track/proximity immediately after - should be blocked by per-track cooldown (2.0s)
        mock_time.return_value = 1.2
        result4 = narrator._should_announce(event, 1.2)
        print(f"Time 1.2 - should announce: {result4}")
        assert result4 is False, "Should be blocked by per-track cooldown"

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
        result5 = narrator._should_announce(event_med, 2.2)
        print(f"Time 2.2 (different proximity) - should announce: {result5}")
        assert result5 is True, f"Different proximity should be allowed after global cooldown, got {result5}"

        # Wait for per-track cooldown to expire
        mock_time.return_value = 3.2
        # Same proximity again - should now be allowed
        result6 = narrator._should_announce(event, 3.2)
        print(f"Time 3.2 (same proximity) - should announce: {result6}")
        assert result6 is True, f"Same proximity should be allowed after per-track cooldown expires, got {result6}"

        print("✓ _should_announce logic works correctly!")

if __name__ == "__main__":
    try:
        test_global_cooldown_logic()
        test_track_reappearance_logic()
        test_should_announce_logic()
        print("\n🎉 All logic tests PASSED! The fixes are working correctly.")
    except Exception as e:
        print(f"\n❌ Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)