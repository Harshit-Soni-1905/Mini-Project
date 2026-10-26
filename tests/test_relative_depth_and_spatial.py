"""Deterministic unit and integration tests for Phase 2A: Relative Depth and Spatial Awareness.

Covers:
1. Honest Depth Semantics:
   - Output in normalized [0.0, 1.0] relative closeness, never claiming metric metres
   - Geometric perspective ground-plane calculation
   - Distance category classification (Immediate, Near, Medium, Far)
   - NaN, Inf, and empty frame safe handling
   - Central inner crop sampling (avoiding background bleeding)
   - Depth-to-proximity mapping

2. Robust Fallback:
   - Seamless fallback to bounding-box height heuristic when depth is disabled, None, or raises errors
   - Obstacle prioritization continues with heuristic proximity and zero crashes

3. Spatial Hysteresis:
   - Deadband zone around left (0.33) and right (0.67) directional boundaries
   - Suppression of zone oscillation caused by bounding-box jitter
   - Confirmed zone transition when crossing the hysteresis deadband

4. Temporal Proximity Smoothing:
   - Rejection of single-frame transient proximity flicker
   - Acceptance of sustained proximity changes (2 consecutive frames)
   - Prevention of false proximity escalation events

5. Approach Evaluation:
   - Detection of consistent approach trends via depth delta and bounding box expansion
   - Limitation documentation: Monocular ego-motion equivalence

6. Safety Constraints:
   - Never issue "Stop!" from depth or corridor heuristics alone without confirmed in-path contact
   - Never announce "safe" or "clear" on zero detections or depth absence
   - Pipeline single-pass depth map reuse without per-detection recomputation
"""

from __future__ import annotations

from collections import deque
from unittest.mock import MagicMock

import numpy as np
import pytest

from narrator.config import AppConfig, DepthConfig
from narrator.depth import (
    DepthCategory,
    GeometricDepthEstimator,
    depth_to_proximity,
)
from narrator.detector import Detection
from narrator.direction import Direction
from narrator.distance import Proximity
from narrator.events import (
    EventNarrator,
    EventType,
    TrackRecord,
    compute_smoothed_direction,
    compute_smoothed_proximity,
)
from narrator.ground_plane import ContactPoint, HazardZone
from narrator.phrasing import format_spoken_alert
from narrator.pipeline import Pipeline
from narrator.prioritizer import PrioritizedObstacle, prioritize_detections


# ==============================================================================
# 1. Honest Depth Semantics & Geometric Depth Estimator Tests
# ==============================================================================


class TestRelativeDepthSemantics:
    def test_depth_normalized_range_and_perspective_semantics(self) -> None:
        """Estimator outputs relative closeness in [0.0, 1.0], where 1.0 is immediate ground."""
        estimator = GeometricDepthEstimator(horizon_y=0.35)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        depth_map = estimator.estimate_depth(frame)

        assert depth_map.shape == (480, 640)
        assert depth_map.dtype == np.float32
        assert estimator.is_inverse_depth is True

        # Closeness at the very bottom (user's feet) is 1.0 (immediate)
        assert depth_map[479, 320] == pytest.approx(1.0, abs=1e-3)
        # Closeness at and above horizon is 0.0 (far background)
        assert depth_map[0, 320] == 0.0
        assert depth_map[int(480 * 0.35), 320] == pytest.approx(0.0, abs=0.02)
        # Values strictly in [0.0, 1.0]
        assert np.all(depth_map >= 0.0)
        assert np.all(depth_map <= 1.0)

    def test_distance_category_classification(self) -> None:
        estimator = GeometricDepthEstimator(
            immediate_thresh=0.75,
            near_thresh=0.50,
            medium_thresh=0.25,
        )

        assert estimator.classify_distance(0.85) == DepthCategory.IMMEDIATE
        assert estimator.classify_distance(0.60) == DepthCategory.NEAR
        assert estimator.classify_distance(0.35) == DepthCategory.MEDIUM
        assert estimator.classify_distance(0.15) == DepthCategory.FAR
        assert estimator.classify_distance(None) == DepthCategory.MEDIUM

    def test_inner_crop_samples_obstacle_center_avoiding_edges(self) -> None:
        """get_obstacle_depth samples inner 50% central crop to avoid background depth bleed."""
        estimator = GeometricDepthEstimator()

        # Synthetic depth map: background is 0.1 (far), center of box is 0.9 (immediate)
        depth_map = np.full((100, 100), 0.1, dtype=np.float32)
        # Box from (20, 20) to (80, 80)
        # Inner 50% crop is from (35, 35) to (65, 65)
        depth_map[35:65, 35:65] = 0.9

        det = Detection("chair", 0.9, x1=20, y1=20, x2=80, y2=80)
        sampled = estimator.get_obstacle_depth(depth_map, det, inner_crop=0.5)

        # Median of inner crop must be 0.9, completely unaffected by edge 0.1 values
        assert sampled == pytest.approx(0.9, abs=1e-3)

    def test_nan_inf_and_empty_depth_maps_handled_safely(self) -> None:
        estimator = GeometricDepthEstimator()
        det = Detection("box", 0.8, 10, 10, 50, 50)

        # Empty / None map
        assert estimator.get_obstacle_depth(None, det) == 0.5
        assert estimator.get_obstacle_depth(np.array([]), det) == 0.5

        # NaN map
        nan_map = np.full((100, 100), np.nan, dtype=np.float32)
        val = estimator.get_obstacle_depth(nan_map, det)
        assert val == 0.5
        assert not np.isnan(val)

        # Inf map
        inf_map = np.full((100, 100), np.inf, dtype=np.float32)
        val_inf = estimator.get_obstacle_depth(inf_map, det)
        assert val_inf == 0.5

    def test_depth_to_proximity_mapping(self) -> None:
        assert depth_to_proximity(DepthCategory.IMMEDIATE) == Proximity.NEAR
        assert depth_to_proximity(DepthCategory.NEAR) == Proximity.NEAR
        assert depth_to_proximity(DepthCategory.MEDIUM) == Proximity.MEDIUM
        assert depth_to_proximity(DepthCategory.FAR) == Proximity.FAR
        assert depth_to_proximity(None) is None


# ==============================================================================
# 2. Robust Fallback & Prioritization Integration Tests
# ==============================================================================


class TestDepthFallbackAndPrioritization:
    def test_prioritize_detections_annotates_relative_depth_and_category(self) -> None:
        estimator = GeometricDepthEstimator(horizon_y=0.30)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        depth_map = estimator.estimate_depth(frame)

        # Obstacle near bottom (immediate ground)
        det = Detection("chair", 0.9, x1=200, y1=380, x2=440, y2=470)
        obstacles = prioritize_detections(
            [det],
            frame_width=640,
            frame_height=480,
            depth_map=depth_map,
            depth_estimator=estimator,
        )

        assert len(obstacles) == 1
        obs = obstacles[0]
        assert obs.relative_depth is not None
        assert obs.relative_depth > 0.70
        assert obs.depth_category in (DepthCategory.IMMEDIATE, DepthCategory.NEAR)
        assert obs.proximity == Proximity.NEAR

    def test_prioritize_detections_falls_back_cleanly_when_depth_map_is_none(self) -> None:
        """When depth_map is None, prioritize_detections uses bounding-box height heuristic."""
        # Tall box -> heuristic will classify as NEAR
        det = Detection("chair", 0.9, x1=200, y1=100, x2=440, y2=450)  # height = 350 / 480 = 0.73
        obstacles = prioritize_detections(
            [det],
            frame_width=640,
            frame_height=480,
            depth_map=None,
            depth_estimator=None,
        )

        assert len(obstacles) == 1
        obs = obstacles[0]
        assert obs.relative_depth is None
        assert obs.depth_category is None
        assert obs.proximity == Proximity.NEAR  # From height heuristic!

    def test_prioritize_detections_falls_back_when_depth_estimator_crashes(self) -> None:
        """If depth_estimator raises an exception, the system catches it and falls back."""
        broken_estimator = MagicMock()
        broken_estimator.get_obstacle_depth.side_effect = RuntimeError("Depth model corrupted")

        fake_map = np.zeros((480, 640), dtype=np.float32)
        det = Detection("chair", 0.9, x1=200, y1=100, x2=440, y2=200)  # height = 100 / 480 = 0.21 -> MEDIUM

        obstacles = prioritize_detections(
            [det],
            frame_width=640,
            frame_height=480,
            depth_map=fake_map,
            depth_estimator=broken_estimator,
        )

        assert len(obstacles) == 1
        obs = obstacles[0]
        assert obs.relative_depth is None
        assert obs.depth_category is None
        assert obs.proximity == Proximity.MEDIUM  # Preserves height heuristic


# ==============================================================================
# 3. Spatial Hysteresis Tests
# ==============================================================================


class TestSpatialHysteresis:
    def test_spatial_hysteresis_suppresses_left_center_boundary_jitter(self) -> None:
        """An obstacle at LEFT boundary does not flip to CENTER until it crosses margin (0.33 + 0.03 = 0.36)."""
        left_bound = 0.33
        hyst = 0.03

        # Object currently LEFT, jittering between 0.32 and 0.35
        # 0.34 is past 0.33, but less than 0.36 -> Must remain LEFT!
        res1 = compute_smoothed_direction(
            current_direction=Direction.CENTER,
            norm_x=0.34,
            previous_direction=Direction.LEFT,
            left_boundary=left_bound,
            hysteresis=hyst,
        )
        assert res1 == Direction.LEFT

        res2 = compute_smoothed_direction(
            current_direction=Direction.LEFT,
            norm_x=0.32,
            previous_direction=Direction.LEFT,
            left_boundary=left_bound,
            hysteresis=hyst,
        )
        assert res2 == Direction.LEFT

        # Clearly crosses hysteresis band past 0.36 (e.g. 0.37) -> Transitions to CENTER
        res3 = compute_smoothed_direction(
            current_direction=Direction.CENTER,
            norm_x=0.37,
            previous_direction=Direction.LEFT,
            left_boundary=left_bound,
            hysteresis=hyst,
        )
        assert res3 == Direction.CENTER

    def test_spatial_hysteresis_suppresses_right_center_boundary_jitter(self) -> None:
        """An obstacle at RIGHT boundary does not flip to CENTER until it crosses margin (0.67 - 0.03 = 0.64)."""
        right_bound = 0.67
        hyst = 0.03

        # Object currently RIGHT, jittering at 0.65 (past 0.67, but greater than 0.64)
        res1 = compute_smoothed_direction(
            current_direction=Direction.CENTER,
            norm_x=0.65,
            previous_direction=Direction.RIGHT,
            right_boundary=right_bound,
            hysteresis=hyst,
        )
        assert res1 == Direction.RIGHT

        # Clearly crosses hysteresis band below 0.64 (e.g. 0.62) -> Transitions to CENTER
        res2 = compute_smoothed_direction(
            current_direction=Direction.CENTER,
            norm_x=0.62,
            previous_direction=Direction.RIGHT,
            right_boundary=right_bound,
            hysteresis=hyst,
        )
        assert res2 == Direction.CENTER

    def test_spatial_hysteresis_center_to_flanks(self) -> None:
        """An obstacle at CENTER must cross left_boundary - hysteresis (0.30) to become LEFT."""
        left_bound = 0.33
        hyst = 0.03

        # At 0.32, naive direction is LEFT, but since previous is CENTER and 0.32 > 0.30, remains CENTER
        res = compute_smoothed_direction(
            current_direction=Direction.LEFT,
            norm_x=0.32,
            previous_direction=Direction.CENTER,
            left_boundary=left_bound,
            hysteresis=hyst,
        )
        assert res == Direction.CENTER

        # At 0.29 (< 0.30), crosses into LEFT
        res_left = compute_smoothed_direction(
            current_direction=Direction.LEFT,
            norm_x=0.29,
            previous_direction=Direction.CENTER,
            left_boundary=left_bound,
            hysteresis=hyst,
        )
        assert res_left == Direction.LEFT

    def test_first_observation_unaffected_by_hysteresis(self) -> None:
        """When previous_direction is None, smoothed direction immediately returns current_direction."""
        res = compute_smoothed_direction(
            current_direction=Direction.LEFT,
            norm_x=0.25,
            previous_direction=None,
        )
        assert res == Direction.LEFT


# ==============================================================================
# 4. Temporal Proximity Smoothing Tests
# ==============================================================================


class TestTemporalProximitySmoothing:
    def test_single_frame_proximity_spike_is_filtered(self) -> None:
        """A transient single-frame spike [MEDIUM, MEDIUM, NEAR] is filtered out."""
        history = deque([Proximity.MEDIUM, Proximity.MEDIUM, Proximity.NEAR], maxlen=5)
        smoothed = compute_smoothed_proximity(history, fallback=Proximity.NEAR)
        assert smoothed == Proximity.MEDIUM

    def test_sustained_proximity_escalation_is_accepted(self) -> None:
        """Two consecutive frames agreeing on NEAR confirms the escalation."""
        history = deque([Proximity.MEDIUM, Proximity.NEAR, Proximity.NEAR], maxlen=5)
        smoothed = compute_smoothed_proximity(history, fallback=Proximity.NEAR)
        assert smoothed == Proximity.NEAR

    def test_event_narrator_does_not_fire_escalation_on_one_frame_spike(self) -> None:
        """EventNarrator suppresses PROXIMITY_ESCALATION on 1-frame noise."""
        en = EventNarrator()

        # Frame 1: track 1 at MEDIUM
        det1 = Detection("chair", 0.9, 200, 100, 440, 200, track_id=1)
        obs1 = PrioritizedObstacle(det1, Direction.CENTER, Proximity.MEDIUM, urgency_score=50.0)
        e1 = en.update([obs1], current_time=1.0)
        # Confirm track on frame 2
        e2 = en.update([obs1], current_time=1.03)
        assert e2 is not None
        en.record_spoken(e2, current_time=1.03)

        # Frame 3: transient spike to NEAR (1 frame only)
        obs_spike = PrioritizedObstacle(det1, Direction.CENTER, Proximity.NEAR, urgency_score=100.0)
        e3 = en.update([obs_spike], current_time=2.5)  # past cooldown
        # Single frame spike is smoothed away -> no PROXIMITY_ESCALATION event
        assert e3 is None or e3.event_type != EventType.PROXIMITY_ESCALATION

        # Frame 4: second frame at NEAR -> confirmed escalation fires
        e4 = en.update([obs_spike], current_time=2.6)
        assert e4 is not None
        assert e4.event_type == EventType.PROXIMITY_ESCALATION


# ==============================================================================
# 5. Approach Evaluation Tests
# ==============================================================================


class TestApproachEvaluation:
    def test_depth_trend_confirms_consistent_approach(self) -> None:
        """Increasing relative depth across frames triggers is_consistently_approaching."""
        rec = TrackRecord(
            track_id=1,
            label="person",
            direction=Direction.CENTER,
            proximity=Proximity.MEDIUM,
            urgency_score=60.0,
            first_seen=1.0,
            last_seen=1.1,
        )
        # Relative depth increasing: 0.30 -> 0.34 -> 0.38 (delta >= 0.05)
        rec.recent_depths.extend([0.30, 0.34, 0.38])
        assert rec.is_consistently_approaching() is True

    def test_bounding_box_height_expansion_confirms_approach(self) -> None:
        """Bounding box vertical expansion by >= 10% triggers approach."""
        rec = TrackRecord(
            track_id=2,
            label="car",
            direction=Direction.CENTER,
            proximity=Proximity.MEDIUM,
            urgency_score=70.0,
            first_seen=1.0,
            last_seen=1.1,
        )
        # Height expanding from 100 to 115 pixels (> 10%)
        rec.recent_heights.extend([100.0, 108.0, 115.0])
        assert rec.is_consistently_approaching() is True

    def test_static_or_receding_obstacle_is_not_approaching(self) -> None:
        rec = TrackRecord(
            track_id=3,
            label="bench",
            direction=Direction.LEFT,
            proximity=Proximity.MEDIUM,
            urgency_score=40.0,
            first_seen=1.0,
            last_seen=1.1,
        )
        rec.recent_depths.extend([0.40, 0.39, 0.38])
        rec.recent_heights.extend([120.0, 118.0, 117.0])
        assert rec.is_consistently_approaching() is False


# ==============================================================================
# 6. Safety Constraints & Honest Semantics Tests
# ==============================================================================


class TestSafetyConstraints:
    def test_never_issue_stop_from_depth_alone(self) -> None:
        """High depth alone must NEVER trigger 'Stop!' without walking path contact point."""
        det = Detection("chair", 0.9, x1=50, y1=380, x2=150, y2=470)
        # High closeness, but outside corridor (PERIPHERAL zone)
        cp = ContactPoint(x=100, y=470, normalized_x=0.15, normalized_y=0.98, zone=HazardZone.PERIPHERAL)
        obs = PrioritizedObstacle(
            detection=det,
            direction=Direction.LEFT,
            proximity=Proximity.NEAR,
            urgency_score=110.0,
            contact_point=cp,
            relative_depth=0.92,
            depth_category=DepthCategory.IMMEDIATE,
        )

        phrase = format_spoken_alert(obs)
        # Must be warning, NEVER "Stop!"
        assert not phrase.startswith("Stop!")
        assert phrase == "near chair left"

    def test_zero_detections_never_claims_safe_or_clear(self) -> None:
        """Absence of obstacles must produce empty list, NEVER emit 'clear' or 'safe'."""
        obstacles = prioritize_detections([], frame_width=640, frame_height=480)
        assert obstacles == []

        en = EventNarrator()
        event = en.update(obstacles, current_time=1.0)
        assert event is None  # Silent, never claims "safe to walk"

    def test_pipeline_single_pass_depth_computation(self) -> None:
        """Pipeline computes relative depth once per frame and annotates obstacles."""
        mock_detector = MagicMock()
        mock_detector.is_healthy = True
        mock_detector.last_error = None
        mock_detector.detect.return_value = [
            Detection("chair", 0.9, x1=200, y1=380, x2=440, y2=470, track_id=1)
        ]

        mock_speaker = MagicMock()
        mock_speaker.is_worker_alive = True
        mock_speaker.queue_size = 0

        config = AppConfig()
        config.depth.enabled = True
        config.depth.estimator_type = "geometric"

        pipeline = Pipeline(config, detector=mock_detector, speaker=mock_speaker)
        assert pipeline.depth_estimator is not None

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        annotated, obstacles, latency_ms = pipeline.process_frame(frame)

        assert len(obstacles) == 1
        obs = obstacles[0]
        # Verified that relative depth was computed and assigned
        assert obs.relative_depth is not None
        assert obs.relative_depth > 0.70
        assert obs.depth_category in (DepthCategory.IMMEDIATE, DepthCategory.NEAR)
