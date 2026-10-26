"""Unit tests for Phase 2C.1 visible ground-region estimation.

Tests cover:
1. GroundRegion geometry and perspective trapezoid corridor generation.
2. Missing, None, empty, and degenerate frame failure modes.
3. Confidence scoring and epistemic uncertainty indicators.
4. Obstacle ground-footprint occlusion masking.
5. Integration with ContactPoint hazard zone classification and corridor hysteresis.
6. Pipeline integration and safety constraint enforcement (no false clearance claims).
7. Execution latency benchmarking (< 1 ms budget).
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from narrator.config import AppConfig, GroundRegionConfig
from narrator.detector import Detection
from narrator.direction import Direction
from narrator.distance import Proximity
from narrator.ground_plane import (
    ContactPoint,
    DisabledGroundRegionEstimator,
    GroundRegion,
    GroundRegionEstimator,
    HazardZone,
    PerspectiveGroundRegionEstimator,
    create_ground_estimator,
    estimate_contact_point,
)
from narrator.pipeline import Pipeline
from narrator.prioritizer import PrioritizedObstacle, prioritize_detections


# Helper to generate a realistic synthetic frame with texture
def make_synthetic_frame(
    h: int = 480, w: int = 640, std: float = 40.0
) -> np.ndarray:
    rng = np.random.default_rng(seed=42)
    base = np.full((h, w, 3), 128, dtype=np.uint8)
    noise = rng.normal(0, std, (h, w, 3)).astype(np.int16)
    return np.clip(base + noise, 0, 255).astype(np.uint8)


class TestGroundRegionGeometryAndPerspective:
    """Validate perspective geometry and trapezoid corridor generation."""

    def test_perspective_estimator_creates_valid_geometry(self) -> None:
        estimator = PerspectiveGroundRegionEstimator(
            horizon_y=0.35,
            ground_y_min=0.45,
            corridor_top_left=0.35,
            corridor_top_right=0.65,
            corridor_bottom_left=0.15,
            corridor_bottom_right=0.85,
        )
        frame = make_synthetic_frame(480, 640)
        region = estimator.estimate_ground_region(frame)

        assert region.is_valid is True
        assert region.confidence == 0.80
        assert region.error_reason is None

        # Ground polygon checks
        assert region.polygon.shape == (4, 2)
        expected_y_start = 0.45 * 480  # 216.0
        np.testing.assert_allclose(region.polygon[0], [0.0, expected_y_start])
        np.testing.assert_allclose(region.polygon[1], [640.0, expected_y_start])
        np.testing.assert_allclose(region.polygon[2], [640.0, 480.0])
        np.testing.assert_allclose(region.polygon[3], [0.0, 480.0])

        # Corridor trapezoid checks (narrow top, wide bottom)
        assert region.corridor_polygon.shape == (4, 2)
        np.testing.assert_allclose(region.corridor_polygon[0], [0.35 * 640, expected_y_start])  # (224, 216)
        np.testing.assert_allclose(region.corridor_polygon[1], [0.65 * 640, expected_y_start])  # (416, 216)
        np.testing.assert_allclose(region.corridor_polygon[2], [0.85 * 640, 480.0])              # (544, 480)
        np.testing.assert_allclose(region.corridor_polygon[3], [0.15 * 640, 480.0])              # (96, 480)

        # Mask checks
        assert region.mask is not None
        assert region.mask.shape == (480, 640)
        assert region.mask.dtype == np.uint8

        # Ground coverage fraction should be approximately (1 - 0.45) = 0.55
        assert 0.50 <= region.ground_fraction <= 0.60
        # No obstacles: corridor is 100% visible
        assert region.corridor_ground_fraction == 1.0
        assert region.corridor_occlusion_fraction == 0.0

    def test_is_in_corridor_and_is_in_ground_spatial_queries(self) -> None:
        estimator = PerspectiveGroundRegionEstimator()
        frame = make_synthetic_frame(480, 640)
        region = estimator.estimate_ground_region(frame)

        # Center point in the middle of the floor (320, 350)
        assert region.is_in_ground(320, 350) is True
        assert region.is_in_corridor(320, 350) is True

        # Point near bottom-left corner of the frame (50, 470)
        # In ground (y >= 216), but outside corridor (96 <= x <= 544)
        assert region.is_in_ground(50, 470) is True
        assert region.is_in_corridor(50, 470) is False

        # Point in the upper portion of the frame (320, 100) (ceiling/wall)
        assert region.is_in_ground(320, 100) is False
        assert region.is_in_corridor(320, 100) is False

    def test_disabled_mask_generation_still_returns_analytic_fractions(self) -> None:
        estimator = PerspectiveGroundRegionEstimator(generate_mask=False)
        frame = make_synthetic_frame(480, 640)
        region = estimator.estimate_ground_region(frame)

        assert region.is_valid is True
        assert region.mask is None
        assert region.corridor_ground_fraction == 1.0
        assert pytest.approx(region.ground_fraction, 0.01) == 0.55


class TestGroundRegionFailureModes:
    """Validate graceful degradation and error handling on edge cases."""

    def test_missing_frame_returns_invalid_region(self) -> None:
        estimator = PerspectiveGroundRegionEstimator()
        region = estimator.estimate_ground_region(None)

        assert region.is_valid is False
        assert region.confidence == 0.0
        assert region.mask is None
        assert region.ground_fraction == 0.0
        assert region.corridor_ground_fraction == 0.0
        assert region.error_reason == "missing_frame"
        assert region.is_in_ground(100, 100) is False
        assert region.is_in_corridor(100, 100) is False

    def test_empty_frame_returns_invalid_region(self) -> None:
        estimator = PerspectiveGroundRegionEstimator()
        empty_frame = np.empty((0, 0, 3), dtype=np.uint8)
        region = estimator.estimate_ground_region(empty_frame)

        assert region.is_valid is False
        assert region.confidence == 0.0
        assert region.error_reason == "empty_or_invalid_frame"

    def test_tiny_frame_dimensions_rejected(self) -> None:
        estimator = PerspectiveGroundRegionEstimator()
        tiny_frame = np.zeros((8, 8, 3), dtype=np.uint8)
        region = estimator.estimate_ground_region(tiny_frame)

        assert region.is_valid is False
        assert region.confidence == 0.0
        assert region.error_reason == "frame_dimensions_too_small"

    def test_disabled_ground_region_estimator(self) -> None:
        estimator = DisabledGroundRegionEstimator()
        frame = make_synthetic_frame(480, 640)
        region = estimator.estimate_ground_region(frame)

        assert region.is_valid is False
        assert region.confidence == 0.0
        assert region.error_reason == "ground_estimation_disabled"

    def test_factory_returns_disabled_when_config_disabled(self) -> None:
        cfg = GroundRegionConfig(enabled=False)
        estimator = create_ground_estimator(cfg)
        assert isinstance(estimator, DisabledGroundRegionEstimator)

    def test_factory_returns_disabled_on_unknown_type(self) -> None:
        cfg = GroundRegionConfig(enabled=True, estimator_type="unsupported_quantum")
        estimator = create_ground_estimator(cfg)
        assert isinstance(estimator, DisabledGroundRegionEstimator)

    def test_estimator_catches_unexpected_exception(self) -> None:
        estimator = PerspectiveGroundRegionEstimator()
        frame = make_synthetic_frame(480, 640)

        with patch("cv2.fillPoly", side_effect=RuntimeError("OpenCV GPU out of memory")):
            region = estimator.estimate_ground_region(frame)
            assert region.is_valid is False
            assert region.confidence == 0.0
            assert "estimator_exception" in (region.error_reason or "")


class TestConfidenceAndUncertainty:
    """Validate epistemic confidence scoring and uncertainty detection."""

    def test_normal_textured_frame_has_nominal_confidence(self) -> None:
        estimator = PerspectiveGroundRegionEstimator(base_confidence=0.80)
        frame = make_synthetic_frame(480, 640, std=35.0)
        region = estimator.estimate_ground_region(frame)

        assert region.is_valid is True
        assert region.confidence == 0.80
        assert region.error_reason is None

    def test_solid_black_degenerate_frame_degrades_confidence(self) -> None:
        estimator = PerspectiveGroundRegionEstimator()
        solid_black = np.zeros((480, 640, 3), dtype=np.uint8)
        region = estimator.estimate_ground_region(solid_black)

        assert region.is_valid is True
        assert region.confidence == 0.15
        assert region.error_reason == "low_variance_degenerate_frame"

    def test_solid_white_degenerate_frame_degrades_confidence(self) -> None:
        estimator = PerspectiveGroundRegionEstimator()
        solid_white = np.full((480, 640, 3), 255, dtype=np.uint8)
        region = estimator.estimate_ground_region(solid_white)

        assert region.is_valid is True
        assert region.confidence == 0.15
        assert region.error_reason == "low_variance_degenerate_frame"

    def test_covered_lens_with_sensor_noise_degrades_confidence(self) -> None:
        """Real webcams in total darkness generate CMOS shot/thermal noise with std > 2.0."""
        estimator = PerspectiveGroundRegionEstimator()
        rng = np.random.default_rng(seed=123)
        # Mean brightness ~ 3.5, noise std ~ 3.0 (exceeds std < 2.0 but mean < 10.0)
        noise = rng.normal(3.5, 3.0, (480, 640, 3))
        covered_frame = np.clip(noise, 0, 255).astype(np.uint8)

        region = estimator.estimate_ground_region(covered_frame)

        assert region.is_valid is True
        assert region.confidence == 0.10
        assert region.error_reason == "covered_lens_or_pitch_black"

    def test_overexposed_saturated_glare_degrades_confidence(self) -> None:
        estimator = PerspectiveGroundRegionEstimator()
        rng = np.random.default_rng(seed=456)
        # Saturated glare with sensor noise std > 2.0 and < 5.0, mean > 248.0
        noise = rng.normal(250.0, 3.2, (480, 640, 3))
        glare_frame = np.clip(noise, 0, 255).astype(np.uint8)

        region = estimator.estimate_ground_region(glare_frame)

        assert region.is_valid is True
        assert region.confidence == 0.10
        assert region.error_reason == "saturated_overexposed_frame"


class TestObstacleOcclusionAndFootprints:
    """Validate obstacle subtraction from ground and corridor visible fractions."""

    def test_obstacle_in_corridor_reduces_corridor_ground_fraction(self) -> None:
        estimator = PerspectiveGroundRegionEstimator(account_for_obstacles=True)
        frame = make_synthetic_frame(480, 640)

        # Baseline with no obstacles
        region_clean = estimator.estimate_ground_region(frame, detections=[])
        assert region_clean.corridor_ground_fraction == 1.0

        # Place a detected obstacle squarely inside the walking corridor on the ground
        # Centered at x=320, width 140 (250 to 390), touching ground (y=250 to 450)
        det_chair = Detection(
            x1=250.0,
            y1=250.0,
            x2=390.0,
            y2=450.0,
            confidence=0.90,
            label="chair",
        )
        region_blocked = estimator.estimate_ground_region(frame, detections=[det_chair])

        assert region_blocked.is_valid is True
        # Corridor visible fraction must decrease proportionally
        assert region_blocked.corridor_ground_fraction < region_clean.corridor_ground_fraction
        assert 0.50 <= region_blocked.corridor_ground_fraction <= 0.85
        assert region_blocked.corridor_occlusion_fraction > 0.15
        # Total ground fraction must also decrease
        assert region_blocked.ground_fraction < region_clean.ground_fraction

    def test_elevated_obstacle_does_not_affect_ground_fraction(self) -> None:
        estimator = PerspectiveGroundRegionEstimator(account_for_obstacles=True)
        frame = make_synthetic_frame(480, 640)

        # Obstacle hanging on the ceiling or wall above the ground horizon (y < 216)
        det_clock = Detection(
            x1=280.0,
            y1=50.0,
            x2=360.0,
            y2=150.0,
            confidence=0.85,
            label="clock",
        )
        region = estimator.estimate_ground_region(frame, detections=[det_clock])

        # Ground and corridor visibility should remain 100% unobstructed
        assert region.corridor_ground_fraction == 1.0
        assert region.corridor_occlusion_fraction == 0.0

    def test_peripheral_obstacle_blocks_ground_but_not_corridor(self) -> None:
        estimator = PerspectiveGroundRegionEstimator(account_for_obstacles=True)
        frame = make_synthetic_frame(480, 640)

        # Obstacle far to the right in the peripheral zone (x=590 to 635, y=300 to 460)
        det_bench = Detection(
            x1=590.0,
            y1=300.0,
            x2=635.0,
            y2=460.0,
            confidence=0.88,
            label="bench",
        )
        region = estimator.estimate_ground_region(frame, detections=[det_bench])

        # Corridor remains unobstructed
        assert region.corridor_ground_fraction == 1.0
        # Overall ground fraction is slightly reduced by the bench footprint
        assert region.ground_fraction < 0.55

    def test_account_for_obstacles_disabled_ignores_detections(self) -> None:
        estimator = PerspectiveGroundRegionEstimator(account_for_obstacles=False)
        frame = make_synthetic_frame(480, 640)

        det_large_box = Detection(
            x1=200.0,
            y1=250.0,
            x2=440.0,
            y2=470.0,
            confidence=0.95,
            label="suitcase",
        )
        region = estimator.estimate_ground_region(frame, detections=[det_large_box])

        # When account_for_obstacles=False, obstacle footprints are not masked out
        assert region.corridor_ground_fraction == 1.0


class TestContactPointAndTrapezoidalCorridor:
    """Validate trapezoidal corridor projection in contact point classification."""

    def test_near_footstep_wide_corridor_catches_low_peripheral_hazard(self) -> None:
        # Perspective corridor widens to [0.15 * 640, 0.85 * 640] = [96, 544] at frame bottom (y=480).
        # An obstacle at x=140, y=470 is at normalized_x = 140/640 = 0.218.
        # Under the static rectangular 0.25..0.75 corridor, 0.218 is outside (PERIPHERAL).
        # Under the perspective trapezoid, at y=470 the corridor spans x >= 96, so x=140 is WALKING_PATH!
        estimator = PerspectiveGroundRegionEstimator()
        frame = make_synthetic_frame(480, 640)
        ground_region = estimator.estimate_ground_region(frame)

        det_near_toy = Detection(
            x1=100.0,
            y1=420.0,
            x2=180.0,
            y2=470.0,  # cx = 140.0
            confidence=0.85,
            label="bottle",
        )

        # 1. With ground_region: accurately classified in immediate walking path
        cp_perspective = estimate_contact_point(
            det_near_toy,
            640,
            480,
            ground_region=ground_region,
        )
        assert cp_perspective.zone == HazardZone.WALKING_PATH
        assert cp_perspective.is_on_ground_region is True

        # 2. Without ground_region (fallback rectangular corridor 0.25..0.75):
        # 140 / 640 = 0.218 < 0.25, falls into PERIPHERAL
        cp_static = estimate_contact_point(
            det_near_toy,
            640,
            480,
            ground_region=None,
        )
        assert cp_static.zone == HazardZone.PERIPHERAL

    def test_corridor_hysteresis_preserved_with_trapezoid(self) -> None:
        estimator = PerspectiveGroundRegionEstimator()
        frame = make_synthetic_frame(480, 640)
        ground_region = estimator.estimate_ground_region(frame)

        # Boundary test at frame bottom: left boundary is 96 px (0.15 normalized).
        # Detection at cx = 94 px (normalized_x = 0.1468)
        det_border = Detection(
            x1=84.0,
            y1=440.0,
            x2=104.0,
            y2=475.0,  # cx = 94.0
            confidence=0.85,
            label="shoe",
        )

        # Case 1: Was WALKING_PATH in previous frame -> hysteresis (0.02) extends boundary to 0.13
        # 0.1468 > 0.13 -> stays WALKING_PATH
        cp_held = estimate_contact_point(
            det_border,
            640,
            480,
            previous_zone=HazardZone.WALKING_PATH,
            ground_region=ground_region,
        )
        assert cp_held.zone == HazardZone.WALKING_PATH

        # Case 2: Was PERIPHERAL in previous frame -> hysteresis requires crossing to 0.17
        # 0.1468 < 0.17 -> stays PERIPHERAL
        cp_blocked = estimate_contact_point(
            det_border,
            640,
            480,
            previous_zone=HazardZone.PERIPHERAL,
            ground_region=ground_region,
        )
        assert cp_blocked.zone == HazardZone.PERIPHERAL


class TestPipelineAndSafetyConstraints:
    """Validate pipeline integration, property accessibility, and strict safety rules."""

    def test_pipeline_exposes_last_ground_region(self) -> None:
        cfg = AppConfig()
        cfg.ground.enabled = True
        pipeline = Pipeline(config=cfg)

        frame = make_synthetic_frame(480, 640)
        annotated, obstacles, latency = pipeline.process_frame(frame)

        # Pipeline must expose last_ground_region property
        assert pipeline.last_ground_region is not None
        assert pipeline.last_ground_region.is_valid is True
        assert pipeline.last_ground_region.confidence == 0.80
        assert pipeline.last_ground_region.mask is not None

    def test_pipeline_ground_disabled_behavior(self) -> None:
        cfg = AppConfig()
        cfg.ground.enabled = False
        pipeline = Pipeline(config=cfg)

        frame = make_synthetic_frame(480, 640)
        annotated, obstacles, latency = pipeline.process_frame(frame)

        assert pipeline.last_ground_region is not None
        assert pipeline.last_ground_region.is_valid is False
        assert pipeline.last_ground_region.error_reason == "ground_estimation_disabled"

    def test_ground_estimation_alone_never_declares_path_safe(self) -> None:
        """SAFETY TEST: Verify ground estimation NEVER generates speech declaring a path safe."""
        cfg = AppConfig()
        cfg.ground.enabled = True
        cfg.speaker.enabled = True
        pipeline = Pipeline(config=cfg)

        # Clear synthetic frame with no obstacles
        frame = make_synthetic_frame(480, 640)
        annotated, obstacles, latency = pipeline.process_frame(frame)

        assert pipeline.last_ground_region is not None
        assert pipeline.last_ground_region.corridor_ground_fraction == 1.0

        # In an empty frame, no speech event must be generated claiming path is safe/clear
        event = pipeline.event_narrator.update(obstacles, frame_width=640)
        assert event is None

        # Verify PrioritizedObstacle speech phrases never contain "safe" or "clear"
        det = Detection(x1=300, y1=300, x2=340, y2=400, confidence=0.8, label="chair")
        obsts = prioritize_detections([det], 640, 480, ground_region=pipeline.last_ground_region)
        for obs in obsts:
            assert "safe" not in obs.speech_text.lower()
            assert "clear" not in obs.speech_text.lower()
            assert "metres" not in obs.speech_text.lower()

    def test_estimator_latency_is_sub_millisecond(self) -> None:
        """PERFORMANCE TEST: Ground estimation must execute in < 1.0 ms."""
        estimator = PerspectiveGroundRegionEstimator(
            account_for_obstacles=True, generate_mask=True
        )
        frame = make_synthetic_frame(480, 640)
        det = Detection(x1=250, y1=250, x2=390, y2=450, confidence=0.9, label="chair")

        # Warm up
        for _ in range(5):
            estimator.estimate_ground_region(frame, detections=[det])

        # Benchmark 50 iterations
        n_iters = 50
        t0 = time.perf_counter()
        for _ in range(n_iters):
            estimator.estimate_ground_region(frame, detections=[det])
        avg_latency_ms = ((time.perf_counter() - t0) / n_iters) * 1000.0

        print(f"\nPerspectiveGroundRegionEstimator average latency: {avg_latency_ms:.3f} ms")
        # Must strictly be well below 1.5 ms (typically 0.15 - 0.35 ms)
        assert avg_latency_ms < 1.5, f"Ground estimation too slow: {avg_latency_ms:.3f} ms"

    def test_pipeline_detects_covered_camera_and_recovers_cleanly(self) -> None:
        """Verify pipeline detects covered camera frames and recovers immediately."""
        cfg = AppConfig()
        cfg.ground.enabled = True
        pipeline = Pipeline(config=cfg)

        # 1. Normal textured frame
        normal_frame = make_synthetic_frame(480, 640)
        pipeline.process_frame(normal_frame)
        assert pipeline.last_ground_region is not None
        assert pipeline.last_ground_region.confidence == 0.80
        assert pipeline.last_ground_region.error_reason is None

        # 2. Covered camera lens (dark frame with CMOS sensor noise)
        rng = np.random.default_rng(seed=999)
        noise = rng.normal(3.5, 3.2, (480, 640, 3))
        covered_frame = np.clip(noise, 0, 255).astype(np.uint8)
        pipeline.process_frame(covered_frame)

        assert pipeline.last_ground_region is not None
        assert pipeline.last_ground_region.confidence == 0.10
        assert pipeline.last_ground_region.error_reason == "covered_lens_or_pitch_black"

        # 3. Uncover camera (normal frame resumes)
        pipeline.process_frame(normal_frame)
        assert pipeline.last_ground_region is not None
        assert pipeline.last_ground_region.confidence == 0.80
        assert pipeline.last_ground_region.error_reason is None

    def test_pipeline_hud_annotations_and_zone_tags(self) -> None:
        """Verify explicit [PATH], [PERI], and [ELEV] tags and contact markers."""
        cfg = AppConfig()
        cfg.ground.enabled = True
        pipeline = Pipeline(config=cfg)

        # Mock detector to produce 3 obstacles:
        # 1. Footstep corridor: backpack at bottom-center (cx=320, y2=470)
        det_path = Detection(x1=270, y1=380, x2=370, y2=470, confidence=0.88, label="backpack", track_id=1)
        # 2. Peripheral floor: handbag at bottom-left corner (cx=50, y2=470)
        det_peri = Detection(x1=20, y1=420, x2=80, y2=470, confidence=0.82, label="handbag", track_id=2)
        # 3. Elevated: clock at top-center (cx=320, y2=150)
        det_elev = Detection(x1=290, y1=70, x2=350, y2=150, confidence=0.75, label="clock", track_id=3)

        pipeline.detector.detect = MagicMock(return_value=[det_path, det_peri, det_elev])

        frame = make_synthetic_frame(480, 640)
        annotated, obstacles, latency = pipeline.process_frame(frame)

        assert len(obstacles) == 3
        obs_by_label = {obs.label: obs for obs in obstacles}

        # Check zone classifications
        assert obs_by_label["backpack"].contact_point.zone == HazardZone.WALKING_PATH
        assert obs_by_label["handbag"].contact_point.zone == HazardZone.PERIPHERAL
        assert obs_by_label["clock"].contact_point.zone == HazardZone.ELEVATED

        # Verify annotated frame is generated and has correct shape
        assert annotated.shape == (480, 640, 3)

    def test_pipeline_hud_ground_badge_dynamic_states(self) -> None:
        """Verify HUD badge text and colors for nominal, degraded, and disabled ground states."""
        cfg = AppConfig()
        cfg.ground.enabled = True
        pipeline = Pipeline(config=cfg)

        # 1. Nominal frame -> [GND: ON 80%]
        normal_frame = make_synthetic_frame(480, 640)
        annotated_normal, _, _ = pipeline.process_frame(normal_frame)
        assert pipeline.last_ground_region.confidence == 0.80

        # 2. Degraded frame -> [GND: DEG 10%]
        rng = np.random.default_rng(seed=101)
        noise = rng.normal(3.0, 2.5, (480, 640, 3))
        covered_frame = np.clip(noise, 0, 255).astype(np.uint8)
        annotated_deg, _, _ = pipeline.process_frame(covered_frame)
        assert pipeline.last_ground_region.confidence == 0.10

        # 3. Disabled ground -> [GND: OFF]
        cfg_off = AppConfig()
        cfg_off.ground.enabled = False
        pipeline_off = Pipeline(config=cfg_off)
        annotated_off, _, _ = pipeline_off.process_frame(normal_frame)
        assert pipeline_off.last_ground_region.is_valid is False

