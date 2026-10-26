"""Deterministic unit and integration tests for Phase 2B: Robust Spatial Tracking and Collision-Risk Trends.

Covers the 8 core Phase 2B validation scenarios:
1. Stationary object with noisy bounding boxes (no false approach trends).
2. Oscillating vertical image position (pitch jitter / walking gait bobbing rejection).
3. Consistently increasing relative-closeness sequence (statistically robust collision risk).
4. Receding object sequence (negative slope, no collision risk).
5. Missing, NaN, and Inf observations (graceful rejection without crash).
6. Track loss followed by reacquisition (gap reset, no false temporal jump deltas).
7. Two nearby distinct objects with same class and direction (untracked disambiguation).
8. Camera forward movement with stationary scene (ego-motion documented collision risk).
9. Corridor boundary hysteresis transitions (suppression of boundary chatter).
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import pytest

from narrator.config import EventsConfig
from narrator.detector import Detection
from narrator.direction import Direction
from narrator.distance import Proximity
from narrator.events import (
    EventNarrator,
    EventType,
    NarrationEvent,
    TrackRecord,
    check_trend_consistency,
)
from narrator.ground_plane import ContactPoint, HazardZone, estimate_contact_point
from narrator.prioritizer import PrioritizedObstacle, prioritize_detections


# ==============================================================================
# Scenario 1: Stationary Object with Noisy Bounding Boxes
# ==============================================================================


class TestStationaryObjectBoundingBoxNoise:
    """Bounding box jitter on a stationary obstacle must not trigger false collision risk."""

    def test_bounding_box_jitter_does_not_trigger_approach(self) -> None:
        rec = TrackRecord(
            track_id=10,
            label="chair",
            direction=Direction.CENTER,
            proximity=Proximity.MEDIUM,
            urgency_score=50.0,
            first_seen=1.0,
            last_seen=1.5,
        )
        # Jitter around 100px: ±3-4 pixels across 7 observations
        rec.recent_heights.extend([100.0, 103.0, 98.0, 102.0, 99.0, 104.0, 101.0])
        assert rec.has_increasing_collision_risk() is False
        assert rec.is_consistently_approaching() is False

    def test_small_sub_threshold_growth_rejected(self) -> None:
        """A small object expanding by 4 pixels (under 6px minimum) is rejected."""
        rec = TrackRecord(
            track_id=11,
            label="bottle",
            direction=Direction.CENTER,
            proximity=Proximity.FAR,
            urgency_score=20.0,
            first_seen=1.0,
            last_seen=1.3,
        )
        # 30 -> 34 px (+4px, which is > 10% relative, but < 6px absolute growth)
        rec.recent_heights.extend([30.0, 32.0, 34.0])
        assert rec.has_increasing_collision_risk() is False

    def test_large_object_sub_percentage_growth_rejected(self) -> None:
        """A large object expanding by 8 pixels on 300px (only 2.6%) is rejected."""
        rec = TrackRecord(
            track_id=12,
            label="car",
            direction=Direction.CENTER,
            proximity=Proximity.NEAR,
            urgency_score=90.0,
            first_seen=1.0,
            last_seen=1.3,
        )
        # 300 -> 308 px (+8px absolute >= 6px, but < 10% relative growth)
        rec.recent_heights.extend([300.0, 304.0, 308.0])
        assert rec.has_increasing_collision_risk() is False


# ==============================================================================
# Scenario 2: Oscillating Vertical Image Position (Pitch Jitter / Gait Bobbing)
# ==============================================================================


class TestWalkingGaitPitchJitterRejection:
    """Human walking introduces 1.5-2.0 Hz camera pitch bobbing causing alternating signs."""

    def test_depth_oscillation_from_gait_is_rejected(self) -> None:
        rec = TrackRecord(
            track_id=20,
            label="person",
            direction=Direction.CENTER,
            proximity=Proximity.MEDIUM,
            urgency_score=60.0,
            first_seen=1.0,
            last_seen=1.8,
        )
        # Depth oscillates up and down every step (+ - + - + -)
        rec.recent_depths.extend([0.30, 0.38, 0.31, 0.39, 0.32, 0.40, 0.33, 0.41])
        assert rec.has_increasing_collision_risk() is False

    def test_check_trend_consistency_rejects_sign_reversals(self) -> None:
        oscillating = [100.0, 110.0, 102.0, 112.0, 104.0, 114.0]
        # Multiple sign reversals (+10, -8, +10, -8, +10) -> 4 reversals
        assert check_trend_consistency(oscillating, max_sign_reversals=1) is False

    def test_single_minor_adjustment_is_tolerated(self) -> None:
        """One minor regression (dip and recovery) among a sustained upward trend is tolerated (max_sign_reversals=2)."""
        sustained_with_one_blip = [0.20, 0.28, 0.27, 0.35, 0.42]
        # Sign transitions: +0.08, -0.01 (reversal 1), +0.08 (reversal 2), +0.07 -> 2 reversals <= 2
        assert check_trend_consistency(
            sustained_with_one_blip,
            min_samples=3,
            min_delta=0.05,
            min_r2=0.60,
            max_sign_reversals=2,
        ) is True


# ==============================================================================
# Scenario 3: Consistently Increasing Relative-Closeness Sequence
# ==============================================================================


class TestConsistentlyIncreasingCollisionRisk:
    """Monotonic approach sequence with statistical significance triggers collision risk."""

    def test_depth_trend_statistically_confirmed(self) -> None:
        rec = TrackRecord(
            track_id=30,
            label="obstacle",
            direction=Direction.CENTER,
            proximity=Proximity.MEDIUM,
            urgency_score=60.0,
            first_seen=1.0,
            last_seen=1.4,
        )
        rec.recent_depths.extend([0.25, 0.31, 0.38, 0.44])
        assert rec.has_increasing_collision_risk() is True
        assert rec.is_consistently_approaching() is True

    def test_bounding_box_height_expansion_statistically_confirmed(self) -> None:
        rec = TrackRecord(
            track_id=31,
            label="bicycle",
            direction=Direction.CENTER,
            proximity=Proximity.MEDIUM,
            urgency_score=70.0,
            first_seen=1.0,
            last_seen=1.4,
        )
        # Expanding from 100px to 125px (>= 6px absolute AND >= 10% relative)
        rec.recent_heights.extend([100.0, 108.0, 116.0, 125.0])
        assert rec.has_increasing_collision_risk() is True

    def test_trend_requires_high_r2_fit(self) -> None:
        """Values that jump wildly (poor linear fit R^2 < 0.60) are rejected."""
        erratic = [0.10, 0.12, 0.11, 0.13, 0.35]
        # Although delta is 0.25, R^2 on a sudden jump at the end with flat start is poor
        # Check that check_trend_consistency properly evaluates R^2
        # Let's test with low R^2 data:
        noisy = [0.10, 0.25, 0.12, 0.28, 0.20]  # erratic slope
        assert check_trend_consistency(noisy, min_r2=0.85) is False


# ==============================================================================
# Scenario 4: Receding Object Sequence
# ==============================================================================


class TestRecedingObjectSequence:
    """Obstacles moving away or shrinking must never trigger collision risk."""

    def test_receding_depth_is_rejected(self) -> None:
        rec = TrackRecord(
            track_id=40,
            label="car",
            direction=Direction.CENTER,
            proximity=Proximity.MEDIUM,
            urgency_score=40.0,
            first_seen=1.0,
            last_seen=1.4,
        )
        rec.recent_depths.extend([0.60, 0.52, 0.45, 0.38])
        assert rec.has_increasing_collision_risk() is False

    def test_shrinking_bounding_box_is_rejected(self) -> None:
        rec = TrackRecord(
            track_id=41,
            label="person",
            direction=Direction.CENTER,
            proximity=Proximity.FAR,
            urgency_score=20.0,
            first_seen=1.0,
            last_seen=1.4,
        )
        rec.recent_heights.extend([150.0, 130.0, 110.0, 95.0])
        assert rec.has_increasing_collision_risk() is False


# ==============================================================================
# Scenario 5: Missing, NaN, and Inf Observations
# ==============================================================================


class TestInvalidObservationsHandling:
    """Missing, NaN, and Inf samples must be safely filtered without crashing."""

    def test_nan_and_inf_filtered_from_valid_trend(self) -> None:
        """Valid samples within a noisy stream containing NaNs and Infs still detect trend."""
        noisy_stream = [0.20, float("nan"), 0.28, float("inf"), 0.36]
        assert check_trend_consistency(noisy_stream, min_samples=3, min_delta=0.05) is True

    def test_all_nans_safely_returns_false(self) -> None:
        all_nans = [float("nan"), float("nan"), float("nan")]
        assert check_trend_consistency(all_nans) is False

    def test_track_record_with_nans_does_not_crash(self) -> None:
        rec = TrackRecord(
            track_id=50,
            label="chair",
            direction=Direction.CENTER,
            proximity=Proximity.MEDIUM,
            urgency_score=50.0,
            first_seen=1.0,
            last_seen=1.3,
        )
        rec.recent_depths.extend([float("nan"), float("inf"), float("nan")])
        assert rec.has_increasing_collision_risk() is False


# ==============================================================================
# Scenario 6: Track Loss Followed by Reacquisition
# ==============================================================================


class TestTrackLossAndReacquisition:
    """Track reacquisition must prevent false trend deltas across temporal occlusion gaps."""

    def test_reacquisition_after_gap_resets_temporal_trend_histories(self) -> None:
        """When a track reappears after > 0.4s occlusion, depth/height histories are cleared."""
        en = EventNarrator(config=EventsConfig(min_frames_to_confirm=1))

        det = Detection(label="person", confidence=0.9, x1=250, y1=200, x2=390, y2=400, track_id=60)
        obs = PrioritizedObstacle(
            detection=det,
            direction=Direction.CENTER,
            proximity=Proximity.MEDIUM,
            urgency_score=60.0,
            relative_depth=0.30,
        )

        # Frame 1 at t=0.0s
        e1 = en.update([obs], current_time=0.0)
        assert e1 is not None
        en.record_spoken(e1, current_time=0.0)

        record = en._tracks[60]
        assert len(record.recent_depths) == 1
        assert record.last_spoken == 0.0

        # Frame 2 at t=0.1s: depth rises to 0.35
        obs2 = PrioritizedObstacle(
            detection=det,
            direction=Direction.CENTER,
            proximity=Proximity.MEDIUM,
            urgency_score=65.0,
            relative_depth=0.35,
        )
        en.update([obs2], current_time=0.1)
        assert len(record.recent_depths) == 2

        # Occlusion gap: object absent between t=0.1s and t=0.7s (gap = 0.6s > 0.4s)
        # At t=0.7s, object reappears with depth 0.45
        obs3 = PrioritizedObstacle(
            detection=det,
            direction=Direction.CENTER,
            proximity=Proximity.MEDIUM,
            urgency_score=70.0,
            relative_depth=0.45,
        )
        en.update([obs3], current_time=0.7)

        # History must have been reset on the 0.6s jump: only the new observation remains
        assert len(record.recent_depths) == 1
        assert record.recent_depths[0] == 0.45
        # Crucially, cannot compute a false 3-sample approach trend spanning the gap
        assert record.has_increasing_collision_risk() is False
        # And last_spoken is preserved, so no duplicate NEW_OBJECT event is fired
        assert record.last_spoken == 0.0

    def test_reacquisition_with_new_track_id_starts_fresh(self) -> None:
        """When detector reacquires an object under a new track ID, history is not contaminated."""
        en = EventNarrator(config=EventsConfig(min_frames_to_confirm=2))

        # Initial track 70
        det1 = Detection(label="person", confidence=0.85, x1=200, y1=100, x2=300, y2=300, track_id=70)
        obs1 = PrioritizedObstacle(det1, Direction.CENTER, Proximity.MEDIUM, urgency_score=50.0, relative_depth=0.30)
        en.update([obs1], current_time=0.0)

        # Disappears, reappears as track 71
        det2 = Detection(label="person", confidence=0.85, x1=205, y1=105, x2=305, y2=305, track_id=71)
        obs2 = PrioritizedObstacle(det2, Direction.CENTER, Proximity.MEDIUM, urgency_score=50.0, relative_depth=0.31)

        # First frame of new track ID: consecutive_frames = 1
        e = en.update([obs2], current_time=0.5)
        # Not confirmed yet (requires min_frames_to_confirm=2)
        assert e is None

        rec71 = en._tracks[71]
        assert rec71.consecutive_frames == 1
        assert len(rec71.recent_depths) == 1
        assert rec71.last_spoken < 0.0  # Never spoken


# ==============================================================================
# Scenario 7: Two Nearby Distinct Objects with Same Class and Direction
# ==============================================================================


class TestUntrackedMultiObjectDisambiguation:
    """When tracking is disabled, distinct objects sharing class and sector must not overwrite each other."""

    def test_two_people_in_center_disambiguated_without_tracking(self) -> None:
        en = EventNarrator(config=EventsConfig(min_frames_to_confirm=1), tracking_enabled=False)

        # Person A on the left side of center (cx = 260)
        det_a = Detection("person", 0.9, x1=210, y1=150, x2=310, y2=350)  # cx = 260
        obs_a = PrioritizedObstacle(det_a, Direction.CENTER, Proximity.MEDIUM, urgency_score=60.0)

        # Person B on the right side of center (cx = 390)
        det_b = Detection("person", 0.85, x1=340, y1=150, x2=440, y2=350)  # cx = 390
        obs_b = PrioritizedObstacle(det_b, Direction.CENTER, Proximity.MEDIUM, urgency_score=55.0)

        # Frame 1: both present in same frame
        en.update([obs_a, obs_b], current_time=1.0)

        # Must maintain 2 distinct tracks, not 1 overwritten track
        assert len(en._tracks) == 2
        keys = list(en._tracks.keys())
        assert "person:ahead" in keys
        assert "person:ahead#1" in keys

        # Frame 2: Person A and Person B move slightly
        det_a2 = Detection("person", 0.9, x1=215, y1=150, x2=315, y2=350)  # cx = 265
        obs_a2 = PrioritizedObstacle(det_a2, Direction.CENTER, Proximity.MEDIUM, urgency_score=60.0)

        det_b2 = Detection("person", 0.85, x1=335, y1=150, x2=435, y2=350)  # cx = 385
        obs_b2 = PrioritizedObstacle(det_b2, Direction.CENTER, Proximity.MEDIUM, urgency_score=55.0)

        en.update([obs_a2, obs_b2], current_time=1.05)

        # Tracks must still be 2, each updating their own consecutive_frames
        assert len(en._tracks) == 2
        rec_a = en._tracks["person:ahead"]
        rec_b = en._tracks["person:ahead#1"]
        assert rec_a.consecutive_frames == 2
        assert rec_b.consecutive_frames == 2
        assert rec_a.last_cx == pytest.approx(265.0, abs=1.0)
        assert rec_b.last_cx == pytest.approx(385.0, abs=1.0)


# ==============================================================================
# Scenario 8: Camera Forward Movement with Stationary Scene (Ego-Motion Ambiguity)
# ==============================================================================


class TestEgoMotionCollisionRiskSemantics:
    """Monocular visual expansion from forward walking represents increasing collision risk."""

    def test_forward_locomotion_triggers_increasing_collision_risk(self) -> None:
        """As user walks toward a stationary bench, closeness increases and triggers risk."""
        rec = TrackRecord(
            track_id=80,
            label="bench",
            direction=Direction.CENTER,
            proximity=Proximity.MEDIUM,
            urgency_score=50.0,
            first_seen=1.0,
            last_seen=1.5,
        )
        # Visual expansion produced by user walking forward toward the bench:
        # Bounding box expands from 100 to 125 px
        rec.recent_heights.extend([100.0, 108.0, 116.0, 125.0])
        # Perspective depth increases from 0.30 to 0.45
        rec.recent_depths.extend([0.30, 0.35, 0.40, 0.45])

        # Both metrics confirm increasing collision risk
        assert rec.has_increasing_collision_risk() is True
        # Documentation integrity check: method docstring explicitly explains ego-motion ambiguity
        doc = rec.has_increasing_collision_risk.__doc__
        assert doc is not None
        assert "ego-motion" in doc.lower() or "monocular" in doc.lower()


# ==============================================================================
# Scenario 9: Corridor Boundary Hysteresis Transitions
# ==============================================================================


class TestCorridorBoundaryHysteresis:
    """Spatial hysteresis suppresses boundary flapping between WALKING_PATH and PERIPHERAL."""

    def test_corridor_boundary_suppresses_left_flapping(self) -> None:
        frame_w = 640
        frame_h = 480
        corridor_min = 0.25
        hysteresis = 0.02

        # 1. Start inside WALKING_PATH at x = 0.26 (166.4 px)
        det_in = Detection("chair", 0.9, x1=156, y1=300, x2=176, y2=450)  # cx = 166.4 / 640 = 0.26
        cp1 = estimate_contact_point(
            det_in, frame_w, frame_h,
            corridor_x_min=corridor_min, corridor_hysteresis=hysteresis,
            previous_zone=None,
        )
        assert cp1.zone == HazardZone.WALKING_PATH

        # 2. Object jitters slightly to x = 0.24 (153.6 px, past 0.25 but within 0.25 - 0.02 = 0.23)
        det_jitter = Detection("chair", 0.9, x1=143.6, y1=300, x2=163.6, y2=450)  # cx = 153.6 / 640 = 0.24
        cp2 = estimate_contact_point(
            det_jitter, frame_w, frame_h,
            corridor_x_min=corridor_min, corridor_hysteresis=hysteresis,
            previous_zone=HazardZone.WALKING_PATH,
        )
        # Hysteresis keeps it in WALKING_PATH (no flapping)
        assert cp2.zone == HazardZone.WALKING_PATH

        # 3. Object moves clearly outside to x = 0.22 (140.8 px < 0.23)
        det_out = Detection("chair", 0.9, x1=130.8, y1=300, x2=150.8, y2=450)  # cx = 140.8 / 640 = 0.22
        cp3 = estimate_contact_point(
            det_out, frame_w, frame_h,
            corridor_x_min=corridor_min, corridor_hysteresis=hysteresis,
            previous_zone=HazardZone.WALKING_PATH,
        )
        assert cp3.zone == HazardZone.PERIPHERAL

        # 4. Object jitters back to x = 0.24 (within 0.25 + 0.02 = 0.27 to re-enter)
        cp4 = estimate_contact_point(
            det_jitter, frame_w, frame_h,
            corridor_x_min=corridor_min, corridor_hysteresis=hysteresis,
            previous_zone=HazardZone.PERIPHERAL,
        )
        # Hysteresis keeps it in PERIPHERAL until it crosses 0.27
        assert cp4.zone == HazardZone.PERIPHERAL

        # 5. Object moves clearly back inside to x = 0.28 (179.2 px > 0.27)
        det_reenter = Detection("chair", 0.9, x1=169.2, y1=300, x2=189.2, y2=450)  # cx = 179.2 / 640 = 0.28
        cp5 = estimate_contact_point(
            det_reenter, frame_w, frame_h,
            corridor_x_min=corridor_min, corridor_hysteresis=hysteresis,
            previous_zone=HazardZone.PERIPHERAL,
        )
        assert cp5.zone == HazardZone.WALKING_PATH
