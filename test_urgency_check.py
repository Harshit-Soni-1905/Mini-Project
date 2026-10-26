#!/usr/bin/env python3
"""Quick verification script for urgency escalation test."""

import sys
from unittest.mock import patch
from src.narrator.events import EventNarrator, EventsConfig
from src.narrator.detector import Detection
from src.narrator.prioritizer import PrioritizedObstacle
from src.narrator.direction import Direction
from src.narrator.distance import Proximity
from src.narrator.events import EventType

def test_urgency_escalation_event():
    """Test that urgency escalation event is triggered correctly."""
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
        det_high = Detection(label="chair", confidence=0.95, x1=50, y1=100, x2=150, y2=200, track_id=22)
        obstacle_high = PrioritizedObstacle(
            detection=det_high,
            direction=Direction.LEFT,
            proximity=Proximity.FAR,
            urgency_score=50.0,  # > 1.3 * 20.0 = 26.0
        )
        mock_time.return_value = 5.0  # 5.0 - 0.1 = 4.9 > 4.0 (track cooldown)
        events = narrator.update([obstacle_high], current_time=5.0)
        print(f"Events returned: {events}")
        if events is not None:
            print(f"Event type: {events.event_type}")
            print(f"Expected: {EventType.URGENCY_ESCALATION}")
            print(f"Test result: {events.event_type == EventType.URGENCY_ESCALATION}")
        else:
            print("No events returned - TEST FAILS")
            return False

        assert events is not None
        assert events.event_type == EventType.URGENCY_ESCALATION
        assert events.label == "chair"
        print("Urgency escalation test PASSED")
        return True

if __name__ == "__main__":
    try:
        test_urgency_escalation_event()
    except Exception as e:
        print(f"Test failed with exception: {e}")
        import traceback
        traceback.print_exc()