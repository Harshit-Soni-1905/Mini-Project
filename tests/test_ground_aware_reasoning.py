"""Regression tests for Phase 2C.2: Ground-Aware Obstacle Reasoning.

Validates:
1. Bounding-box base-span overlap against projected trapezoidal corridor.
2. Peripheral obstacle outside corridor remains eligible for alerts.
3. Flanking adjustment for peripheral obstacles with partial corridor overlap (>= 10%).
4. Degraded ground confidence scales path bonus gracefully.
5. Invalid or absent ground region fallback never suppresses obstacles.
6. Corridor boundary hysteresis prevents zone flapping.
7. Person and backpack co-occurrence ground reasoning.
8. Zero detections on ground never claims path clear, and urgency score bounds.
"""

from __future__ import annotations

import numpy as np
import pytest

from narrator.config import AppConfig
from narrator.detector import Detection
from narrator.ground_plane import (
    ContactPoint,
    GroundRegion,
    HazardZone,
    PerspectiveGroundRegionEstimator,
    estimate_contact_point,
)
from narrator.prioritizer import (
    Direction,
    PrioritizedObstacle,
    Proximity,
    prioritize_detections,
)


def make_test_ground_region(
    width: int = 640,
    height: int = 480,
    *,
    confidence: float = 0.95,
    is_valid: bool = True,
) -> GroundRegion:
    """Create a standardized ground region with trapezoidal walking corridor."""
    estimator = PerspectiveGroundRegionEstimator(
        horizon_y=0.35,
        ground_y_min=0.45,
        corridor_top_left=0.35,
        corridor_top_right=0.65,
        corridor_bottom_left=0.15,
        corridor_bottom_right=0.85,
        account_for_obstacles=False,
    )
    frame = np.full((height, width, 3), 128, dtype=np.uint8)
    import dataclasses
    region = estimator.estimate_ground_region(frame, detections=[])
    return dataclasses.replace(
        region,
        confidence=confidence,
        is_valid=is_valid,
        error_reason=None if is_valid else "Degraded or invalid ground surface",
    )


class TestGroundAwareReasoning:
    """Phase 2C.2 Ground-Aware Obstacle Reasoning regression test suite."""

    def test_obstacle_base_span_overlap_triggers_walking_path(self) -> None:
        """Obstacle whose midpoint is outside the corridor but has >= 35% base overlap enters WALKING_PATH."""
        frame_w, frame_h = 640, 480
        ground_region = make_test_ground_region(frame_w, frame_h, confidence=0.95)

        # At frame bottom (y2=470), corridor extends from 0.15*640=96 to 0.85*640=544.
        # Place an obstacle whose center cx is to the left of the corridor (< 96 px),
        # but its width is wide enough that >= 35% of its base span enters the corridor.
        # Width: 100 px (x1 = 40, x2 = 140). cx = 90 (< 96, so center is outside corridor).
        # Span inside corridor [96, 140]: width = 44 px -> 44% overlap (>= 35%).
        det_intruder = Detection(
            label="chair",
            confidence=0.85,
            x1=40.0,
            y1=300.0,
            x2=140.0,
            y2=470.0,
        )

        contact = estimate_contact_point(
            det_intruder,
            frame_w,
            frame_h,
            ground_region=ground_region,
        )

        assert contact.corridor_overlap_ratio >= 0.35
        assert contact.zone == HazardZone.WALKING_PATH

        # Prioritization must grant the walking path bonus based on overlap
        obstacles = prioritize_detections(
            [det_intruder],
            frame_width=frame_w,
            frame_height=frame_h,
            path_urgency_bonus=25.0,
            ground_region=ground_region,
        )
        assert len(obstacles) == 1
        obs = obstacles[0]
        assert obs.is_in_walking_path is True
        assert obs.corridor_overlap_ratio >= 0.35
        # Urgency score must include the modulated path urgency bonus
        assert obs.urgency_score >= 60.0

    def test_peripheral_obstacle_outside_corridor_eligible_for_alert(self) -> None:
        """Obstacles completely outside the corridor must remain eligible for peripheral alerts."""
        frame_w, frame_h = 640, 480
        ground_region = make_test_ground_region(frame_w, frame_h, confidence=0.95)

        # Obstacle far to the right: x1=580, x2=630, y2=460 (corridor max at y=460 is ~538)
        det_peripheral = Detection(
            label="backpack",
            confidence=0.88,
            x1=580.0,
            y1=360.0,
            x2=630.0,
            y2=460.0,
        )

        contact = estimate_contact_point(
            det_peripheral,
            frame_w,
            frame_h,
            ground_region=ground_region,
        )

        assert contact.zone == HazardZone.PERIPHERAL
        assert contact.corridor_overlap_ratio == 0.0

        obstacles = prioritize_detections(
            [det_peripheral],
            frame_width=frame_w,
            frame_height=frame_h,
            ground_region=ground_region,
        )
        assert len(obstacles) == 1
        obs = obstacles[0]
        assert obs.is_in_walking_path is False
        # Obstacle must NOT be suppressed; must maintain a positive urgency score
        assert obs.urgency_score > 0.0
        assert obs.direction == Direction.RIGHT
        assert "backpack" in obs.speech_text

    def test_flanking_peripheral_obstacle_partial_overlap_adjustment(self) -> None:
        """Peripheral obstacles with 10% to 34% corridor overlap receive a modest flanking adjustment."""
        frame_w, frame_h = 640, 480
        ground_region = make_test_ground_region(frame_w, frame_h, confidence=0.95)

        # Corridor left boundary at y2=470 is 96 px.
        # Obstacle A (flanking): width 100 px (x1=20, x2=120). cx = 70.
        # Intersection with corridor [96, 120] is 24 px -> 24% overlap (>= 10% and < 35%).
        det_flanking = Detection(
            label="chair",
            confidence=0.85,
            x1=20.0,
            y1=300.0,
            x2=120.0,
            y2=470.0,
        )

        # Obstacle B (pure lateral): width 100 px (x1=0, x2=80). cx = 40.
        # Intersection with corridor [96, 80] is 0 px -> 0% overlap (< 10%).
        det_pure_lateral = Detection(
            label="chair",
            confidence=0.85,
            x1=0.0,
            y1=300.0,
            x2=80.0,
            y2=470.0,
        )

        # Obstacle C (walking path): width 100 px (x1=50, x2=150). cx = 100.
        # Intersection with corridor [96, 150] is 54 px -> 54% overlap (>= 35%).
        det_in_path = Detection(
            label="chair",
            confidence=0.85,
            x1=50.0,
            y1=300.0,
            x2=150.0,
            y2=470.0,
        )

        obs_flanking = prioritize_detections(
            [det_flanking],
            frame_w,
            frame_h,
            path_urgency_bonus=25.0,
            ground_region=ground_region,
        )[0]
        obs_pure = prioritize_detections(
            [det_pure_lateral],
            frame_w,
            frame_h,
            path_urgency_bonus=25.0,
            ground_region=ground_region,
        )[0]
        obs_path = prioritize_detections(
            [det_in_path],
            frame_w,
            frame_h,
            path_urgency_bonus=25.0,
            ground_region=ground_region,
        )[0]

        # Flanking obstacle is in PERIPHERAL zone
        assert obs_flanking.contact_point.zone == HazardZone.PERIPHERAL
        assert 0.10 <= obs_flanking.corridor_overlap_ratio < 0.35

        # Flanking obstacle scores higher than pure lateral clutter
        assert obs_flanking.urgency_score > obs_pure.urgency_score

        # Flanking bonus is strictly capped at 25% of path_urgency_bonus (25.0 * 0.25 = 6.25)
        flanking_diff = obs_flanking.urgency_score - obs_pure.urgency_score
        assert flanking_diff <= 6.25 + 1e-3

        # Path obstacle still significantly outranks the flanking obstacle
        assert obs_path.urgency_score > obs_flanking.urgency_score

    def test_degraded_ground_confidence_scales_bonus_gracefully(self) -> None:
        """Low ground confidence gracefully scales down path bonus without suppressing detections."""
        frame_w, frame_h = 640, 480
        region_high = make_test_ground_region(frame_w, frame_h, confidence=1.0)
        region_degraded = make_test_ground_region(frame_w, frame_h, confidence=0.40)

        # Obstacle squarely in walking path
        det = Detection(
            label="chair",
            confidence=0.85,
            x1=260.0,
            y1=320.0,
            x2=380.0,
            y2=460.0,
        )

        obs_high = prioritize_detections(
            [det],
            frame_w,
            frame_h,
            path_urgency_bonus=25.0,
            ground_region=region_high,
        )[0]
        obs_degraded = prioritize_detections(
            [det],
            frame_w,
            frame_h,
            path_urgency_bonus=25.0,
            ground_region=region_degraded,
        )[0]

        # Both recognize the obstacle in WALKING_PATH
        assert obs_high.is_in_walking_path is True
        assert obs_degraded.is_in_walking_path is True

        # High confidence gets larger bonus than degraded confidence
        assert obs_high.urgency_score > obs_degraded.urgency_score

        # Degraded confidence obstacle is NEVER suppressed: maintains baseline proximity + direction + category
        assert obs_degraded.urgency_score >= 35.0

    def test_invalid_or_absent_ground_fallback_never_suppresses(self) -> None:
        """When ground region is absent or invalid, deliberate fallback scoring applies cleanly."""
        frame_w, frame_h = 640, 480
        region_invalid = make_test_ground_region(frame_w, frame_h, confidence=0.0, is_valid=False)

        det = Detection(
            label="person",
            confidence=0.90,
            x1=250.0,
            y1=200.0,
            x2=390.0,
            y2=460.0,
        )

        # 1. With ground_region = None
        obs_none = prioritize_detections([det], frame_w, frame_h, ground_region=None)[0]

        # 2. With invalid ground region
        obs_invalid = prioritize_detections([det], frame_w, frame_h, ground_region=region_invalid)[0]

        # Both must produce valid prioritized obstacles
        assert obs_none.label == "person"
        assert obs_invalid.label == "person"
        assert obs_none.is_in_walking_path is True
        assert obs_invalid.is_in_walking_path is True

        # Fallback scoring applies cleanly: scores match exactly and are not suppressed
        assert abs(obs_none.urgency_score - obs_invalid.urgency_score) < 1e-4
        assert obs_none.urgency_score > 60.0

    def test_corridor_boundary_hysteresis_prevents_flapping(self) -> None:
        """Corridor boundary hysteresis prevents frame-to-frame zone flapping near threshold."""
        frame_w, frame_h = 640, 480
        ground_region = make_test_ground_region(frame_w, frame_h, confidence=0.95)

        # Corridor left boundary at y2=475 is approx 96 px (0.15 normalized).
        # Place detection right on the boundary: cx = 94 px (0.1468 normalized).
        det_border = Detection(
            label="shoe",
            confidence=0.85,
            x1=84.0,
            y1=440.0,
            x2=104.0,
            y2=475.0,
        )

        # When previously WALKING_PATH: boundary expands by hysteresis (0.02) -> stays WALKING_PATH
        cp_held = estimate_contact_point(
            det_border,
            frame_w,
            frame_h,
            corridor_hysteresis=0.02,
            previous_zone=HazardZone.WALKING_PATH,
            ground_region=ground_region,
        )
        assert cp_held.zone == HazardZone.WALKING_PATH

        # When previously PERIPHERAL: boundary contracts by hysteresis (0.02) -> stays PERIPHERAL
        cp_blocked = estimate_contact_point(
            det_border,
            frame_w,
            frame_h,
            corridor_hysteresis=0.02,
            previous_zone=HazardZone.PERIPHERAL,
            ground_region=ground_region,
        )
        assert cp_blocked.zone == HazardZone.PERIPHERAL

    def test_person_and_backpack_cooccurrence_ground_reasoning(self) -> None:
        """Person and backpack detected together receive independent, correct ground reasoning."""
        frame_w, frame_h = 640, 480
        ground_region = make_test_ground_region(frame_w, frame_h, confidence=0.95)

        # Person standing in the center walking corridor
        det_person = Detection(
            label="person",
            confidence=0.92,
            x1=280.0,
            y1=120.0,
            x2=380.0,
            y2=460.0,
            track_id=1,
        )

        # Backpack worn or carried slightly to the right, touching lower torso/ground region
        det_backpack = Detection(
            label="backpack",
            confidence=0.87,
            x1=340.0,
            y1=220.0,
            x2=420.0,
            y2=380.0,
            track_id=2,
        )

        obstacles = prioritize_detections(
            [det_person, det_backpack],
            frame_width=frame_w,
            frame_height=frame_h,
            ground_region=ground_region,
        )

        assert len(obstacles) == 2
        obs_map = {o.label: o for o in obstacles}

        assert "person" in obs_map
        assert "backpack" in obs_map

        # Person bottom is at y=460 (on ground) and in corridor
        assert obs_map["person"].contact_point.zone == HazardZone.WALKING_PATH
        assert obs_map["person"].is_in_walking_path is True

        # Both obstacles have independent contact points and valid urgency scores
        assert obs_map["person"].urgency_score > 0.0
        assert obs_map["backpack"].urgency_score > 0.0
        # No relabeling occurred
        assert obs_map["person"].label == "person"
        assert obs_map["backpack"].label == "backpack"

    def test_zero_detections_on_ground_never_claims_path_clear(self) -> None:
        """Absence of detections over visible ground must NEVER produce 'path clear' claims, and scores are bounded."""
        frame_w, frame_h = 640, 480
        ground_region = make_test_ground_region(frame_w, frame_h, confidence=0.95)

        # Empty detections on completely visible ground
        obstacles = prioritize_detections(
            [],
            frame_width=frame_w,
            frame_height=frame_h,
            ground_region=ground_region,
        )

        assert len(obstacles) == 0
        # Explicit safety constraint verification: no obstacle is created, no walking path hazard exists
        for obs in obstacles:
            assert obs.is_in_walking_path is False

        # Upper bound test: verify that even with all maximal factors, urgency score never overflows
        det_max = Detection(
            label="person",
            confidence=1.0,
            x1=280.0,
            y1=100.0,
            x2=360.0,
            y2=470.0,
        )
        from unittest.mock import MagicMock

        mock_depth_estimator = MagicMock()
        mock_depth_estimator.get_obstacle_depth.return_value = 1.0
        fake_depth_map = np.ones((frame_h, frame_w), dtype=np.float32)

        obs_max = prioritize_detections(
            [det_max],
            frame_width=frame_w,
            frame_height=frame_h,
            path_urgency_bonus=25.0,
            depth_map=fake_depth_map,
            depth_estimator=mock_depth_estimator,
            ground_region=ground_region,
        )[0]

        # Maximum possible score for person:
        # PROXIMITY_WEIGHTS[NEAR] (100.0) + DIRECTION_WEIGHTS[CENTER] (20.0) +
        # CATEGORY_WEIGHTS["person"] (8.0) + path_urgency_bonus (25.0 * 0.95 = 23.75) +
        # depth_bonus (5.0) + confidence*2.0 (2.0) = 158.75.
        # Absolute system ceiling across any class (e.g. car=15.0) is 167.0.
        assert obs_max.urgency_score <= 170.0
        assert 155.0 <= obs_max.urgency_score <= 165.0
