"""Unit tests for Phase 2 (Everyday Object Vocabulary) & Phase 3 (Ground-Path Hazard Assessment).

Covers:
- Vocabulary category groupings (walking hazards, dynamic hazards, landmarks, secondary objects)
- Predefined speech profiles (navigation_only, full_awareness, indoor, outdoor)
- EventsConfig class resolution logic
- Contact point estimation (walking path corridor, peripheral, elevated)
- Information hierarchy: walking path obstacles outranking elevated tabletop items
"""

from __future__ import annotations

import pytest

from narrator.config import (
    DYNAMIC_HAZARD_CLASSES,
    NAVIGATION_LANDMARKS,
    OBSTACLE_CLASSES,
    SECONDARY_OBJECT_CLASSES,
    SPEECH_PROFILES,
    WALKING_HAZARD_CLASSES,
    EventsConfig,
    WalkingCorridorConfig,
)
from narrator.detector import Detection
from narrator.distance import Proximity
from narrator.ground_plane import ContactPoint, HazardZone, estimate_contact_point
from narrator.prioritizer import prioritize_detections


class TestVocabularyCategories:
    def test_walking_hazards_defined(self) -> None:
        assert "chair" in WALKING_HAZARD_CLASSES
        assert "dining table" in WALKING_HAZARD_CLASSES
        assert "bench" in WALKING_HAZARD_CLASSES
        assert "potted plant" in WALKING_HAZARD_CLASSES
        assert "suitcase" in WALKING_HAZARD_CLASSES
        assert "backpack" in WALKING_HAZARD_CLASSES
        assert "handbag" in WALKING_HAZARD_CLASSES

    def test_dynamic_hazards_defined(self) -> None:
        assert "person" in DYNAMIC_HAZARD_CLASSES
        assert "car" in DYNAMIC_HAZARD_CLASSES
        assert "bicycle" in DYNAMIC_HAZARD_CLASSES
        assert "dog" in DYNAMIC_HAZARD_CLASSES

    def test_secondary_objects_defined(self) -> None:
        assert "bottle" in SECONDARY_OBJECT_CLASSES
        assert "cup" in SECONDARY_OBJECT_CLASSES
        assert "laptop" in SECONDARY_OBJECT_CLASSES
        assert "cell phone" in SECONDARY_OBJECT_CLASSES

    def test_speech_profiles_coverage(self) -> None:
        assert "navigation_only" in SPEECH_PROFILES
        assert "full_awareness" in SPEECH_PROFILES
        assert "indoor" in SPEECH_PROFILES
        assert "outdoor" in SPEECH_PROFILES

        # navigation_only excludes small everyday clutter
        nav_classes = SPEECH_PROFILES["navigation_only"]
        assert "chair" in nav_classes
        assert "person" in nav_classes
        assert "bottle" not in nav_classes
        assert "laptop" not in nav_classes

    def test_events_config_profile_resolution(self) -> None:
        # Default full_awareness
        cfg_default = EventsConfig()
        assert cfg_default.resolve_alert_classes() is not None
        assert "bottle" in cfg_default.resolve_alert_classes()

        # navigation_only
        cfg_nav = EventsConfig(speech_profile="navigation_only")
        allowed = cfg_nav.resolve_alert_classes()
        assert "chair" in allowed
        assert "person" in allowed
        assert "bottle" not in allowed

        # Explicit alert_classes overrides speech_profile
        cfg_explicit = EventsConfig(
            speech_profile="navigation_only",
            alert_classes={"person", "door"},
        )
        assert cfg_explicit.resolve_alert_classes() == {"person", "door"}


class TestContactPointAndGroundPlane:
    def test_contact_point_in_walking_path(self) -> None:
        # Detection centered horizontally, bottom touching lower half of frame
        det = Detection("chair", 0.9, x1=260, y1=240, x2=380, y2=450)
        cp = estimate_contact_point(det, frame_width=640, frame_height=480)

        assert cp.zone == HazardZone.WALKING_PATH
        assert cp.normalized_y == 450 / 480  # ~0.9375 >= 0.45
        assert 0.25 <= cp.normalized_x <= 0.75

    def test_contact_point_peripheral(self) -> None:
        # Detection on far right, bottom on ground
        det = Detection("chair", 0.9, x1=550, y1=240, x2=630, y2=450)
        cp = estimate_contact_point(det, frame_width=640, frame_height=480)

        assert cp.zone == HazardZone.PERIPHERAL
        assert cp.normalized_x > 0.75

    def test_contact_point_elevated_tabletop(self) -> None:
        # Bottle resting on a high table / shelf (bottom at y2 = 120 in 480 frame)
        det = Detection("bottle", 0.85, x1=300, y1=60, x2=340, y2=120)
        cp = estimate_contact_point(det, frame_width=640, frame_height=480)

        assert cp.zone == HazardZone.ELEVATED
        assert cp.normalized_y == 120 / 480  # 0.25 < 0.45


class TestInformationHierarchy:
    def test_ground_obstacle_dominates_elevated_secondary_object(self) -> None:
        # Chair in walking path
        chair = Detection("chair", 0.85, x1=250, y1=200, x2=390, y2=460)  # height 260/480 = 0.54 (NEAR), center

        # Bottle on table (elevated)
        bottle = Detection("bottle", 0.95, x1=300, y1=50, x2=340, y2=150)  # height 100/480 = 0.208 (MEDIUM), center

        obstacles = prioritize_detections(
            [bottle, chair],
            frame_width=640,
            frame_height=480,
            path_urgency_bonus=25.0,
        )

        assert obstacles[0].label == "chair"
        assert obstacles[0].is_in_walking_path is True
        assert obstacles[1].label == "bottle"
        assert obstacles[1].is_in_walking_path is False
        assert obstacles[0].urgency_score > obstacles[1].urgency_score + 40.0


class TestBackpackAndLuggageTaxonomyAndPipeline:
    def test_backpack_in_all_speech_profiles(self) -> None:
        """Verify backpack and handbag are present in all functional speech profiles."""
        assert "backpack" in SPEECH_PROFILES["navigation_only"]
        assert "handbag" in SPEECH_PROFILES["navigation_only"]
        assert "backpack" in SPEECH_PROFILES["full_awareness"]
        assert "handbag" in SPEECH_PROFILES["full_awareness"]
        assert "backpack" in SPEECH_PROFILES["indoor"]
        assert "handbag" in SPEECH_PROFILES["indoor"]
        assert "backpack" in SPEECH_PROFILES["outdoor"]
        assert "handbag" in SPEECH_PROFILES["outdoor"]

    def test_backpack_in_walking_path_receives_corridor_bonus(self) -> None:
        """Verify backpack on floor in walking path gets path_urgency_bonus."""
        # Backpack in center corridor on floor
        det_path = Detection("backpack", 0.85, x1=250, y1=300, x2=390, y2=460)
        # Backpack in peripheral zone on floor
        det_peri = Detection("backpack", 0.85, x1=520, y1=300, x2=630, y2=460)

        obstacles = prioritize_detections(
            [det_path, det_peri],
            frame_width=640,
            frame_height=480,
            path_urgency_bonus=25.0,
        )

        assert len(obstacles) == 2
        assert obstacles[0].detection == det_path
        assert obstacles[0].is_in_walking_path is True
        assert obstacles[1].detection == det_peri
        assert obstacles[1].is_in_walking_path is False
        assert obstacles[0].urgency_score >= obstacles[1].urgency_score + 25.0

    def test_co_occurring_person_and_backpack_both_survive(self) -> None:
        """Verify when person and backpack co-occur, both survive in prioritized obstacles."""
        det_person = Detection("person", 0.90, x1=100, y1=50, x2=280, y2=460, track_id=1)
        det_backpack = Detection("backpack", 0.75, x1=350, y1=260, x2=450, y2=460, track_id=2)

        obstacles = prioritize_detections(
            [det_person, det_backpack],
            frame_width=640,
            frame_height=480,
            path_urgency_bonus=25.0,
        )

        assert len(obstacles) == 2
        labels = [o.label for o in obstacles]
        assert "person" in labels
        assert "backpack" in labels
        # Person has higher category weight (8.0 vs 2.0)
        assert obstacles[0].label == "person"
        assert obstacles[1].label == "backpack"

    def test_backpack_alone_triggers_narration_event(self) -> None:
        """Verify standalone backpack detection confirms and generates a spoken event."""
        from src.narrator.events import EventNarrator, EventsConfig, EventType

        config = EventsConfig(min_frames_to_confirm=2)
        narrator = EventNarrator(config=config)

        det_backpack = Detection("backpack", 0.80, x1=260, y1=240, x2=380, y2=450, track_id=10)
        obstacles = prioritize_detections([det_backpack], frame_width=640, frame_height=480)

        # Frame 1: establishes track
        event1 = narrator.update(obstacles, current_time=0.0)
        assert event1 is None

        # Frame 2: confirmed track triggers NEW_OBJECT
        event2 = narrator.update(obstacles, current_time=0.1)
        assert event2 is not None
        assert event2.label == "backpack"
        assert event2.event_type == EventType.NEW_OBJECT
        assert "backpack" in event2.speech_text

    def test_sequential_person_then_backpack_announcement(self) -> None:
        """Verify co-occurring person and backpack are announced sequentially after global cooldown."""
        from src.narrator.events import EventNarrator, EventsConfig

        config = EventsConfig(global_cooldown_seconds=1.5, min_frames_to_confirm=1)
        narrator = EventNarrator(config=config)

        det_person = Detection("person", 0.90, x1=200, y1=50, x2=350, y2=460, track_id=1)
        det_backpack = Detection("backpack", 0.80, x1=400, y1=300, x2=500, y2=460, track_id=2)

        obstacles = prioritize_detections([det_person, det_backpack], frame_width=640, frame_height=480)

        # Frame 1: Person wins first speech slot
        event1 = narrator.update(obstacles, current_time=0.0)
        assert event1 is not None
        assert event1.label == "person"
        narrator.record_spoken(event1, current_time=0.0)

        # Frame 2 (during global cooldown): Speech is throttled
        event2 = narrator.update(obstacles, current_time=0.5)
        assert event2 is None

        # Frame 3 (after global cooldown): Backpack can now be announced
        event3 = narrator.update(obstacles, current_time=2.0)
        assert event3 is not None
        assert event3.label == "backpack"
        narrator.record_spoken(event3, current_time=2.0)

    def test_genuine_person_not_suppressed_or_altered(self) -> None:
        """Verify genuine person detection remains labeled 'person' with original confidence."""
        det_person = Detection("person", 0.92, x1=150, y1=40, x2=320, y2=470, track_id=5)
        obstacles = prioritize_detections([det_person], frame_width=640, frame_height=480)

        assert len(obstacles) == 1
        assert obstacles[0].label == "person"
        assert obstacles[0].detection.confidence == 0.92
        assert obstacles[0].detection.track_id == 5

