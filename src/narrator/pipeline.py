"""Live camera processing pipeline.

Orchestrates frame capture from OpenCV, YOLO detection, directional
and proximity estimation, hazard prioritization, on-screen visualization,
and asynchronous text-to-speech announcement.
"""

from __future__ import annotations

import logging
import time

import cv2
import numpy as np

from narrator.config import AppConfig
from narrator.depth import DepthEstimator, GeometricDepthEstimator
from narrator.detector import Detection, Detector
from narrator.distance import Proximity
from narrator.events import EventNarrator, NarrationEvent
from narrator.ground_plane import (
    GroundRegion,
    GroundRegionEstimator,
    HazardZone,
    create_ground_estimator,
)
from narrator.health import HealthMonitor
from narrator.metrics import PerformanceMonitor
from narrator.prioritizer import PrioritizedObstacle, prioritize_detections
from narrator.speaker import Speaker

logger = logging.getLogger(__name__)

# Debug narration logger
narration_logger = logging.getLogger("narrator.narration")

# Throttled diagnostic logger for stage-by-stage pipeline visibility
diag_logger = logging.getLogger("narrator.diagnostics")
_diag_last_log = 0.0
_diag_log_interval = 2.0  # seconds between diagnostic summaries

def _maybe_log_diagnostics(
    detections: list[Detection],
    obstacles: list[PrioritizedObstacle],
    event: NarrationEvent | None,
    speak_result: bool | None,
    config,
    speaker,
    health: HealthMonitor | None = None,
    submission_latency_ms: float | None = None,
    capture_latency_ms: float = 0.0,
    inference_latency_ms: float = 0.0,
    depth_latency_ms: float = 0.0,
    total_latency_ms: float = 0.0,
    depth_ok: bool = False,
    ground_latency_ms: float = 0.0,
    ground_region: GroundRegion | None = None,
) -> None:
    """Log throttled stage-by-stage diagnostic summary with separated latencies."""
    global _diag_last_log
    import time
    now = time.monotonic()
    if now - _diag_last_log < _diag_log_interval:
        return
    _diag_last_log = now

    # Count tracks with track_id vs spatial-only
    tracked_count = sum(1 for o in obstacles if o.detection.track_id is not None)
    spatial_count = len(obstacles) - tracked_count

    # Event and speech status
    event_info = "None"
    suppression_reason = None
    if event is not None:
        event_info = f"{event.event_type.value}:{event.label}:{event.direction.value}"
        # Check if speaker would have rejected it
        if config.speaker.enabled:
            if not speak_result:
                suppression_reason = "speaker_cooldown_or_queue_full"
    elif config.debug_detections:
        # Log why no event was selected
        suppression_reason = "no_eligible_event"

    health_status = health.get_snapshot().is_healthy if health else "N/A"
    queue_wait_ms = getattr(speaker, "last_queue_latency_ms", 0.0) if speaker else 0.0
    playback_dur_ms = getattr(speaker, "last_playback_duration_ms", 0.0) if speaker else 0.0

    # Ground estimator summary
    gnd_summary = "N/A"
    if ground_region is not None:
        if not ground_region.is_valid:
            gnd_summary = f"invalid({ground_region.error_reason})"
        elif ground_region.confidence < 0.50:
            gnd_summary = f"degraded(conf={ground_region.confidence:.2f},{ground_region.error_reason})"
        else:
            gnd_summary = f"ok(conf={ground_region.confidence:.2f},corr={ground_region.corridor_ground_fraction:.0%})"
    elif not getattr(config.ground, "enabled", True):
        gnd_summary = "disabled"

    diag_logger.info(
        "PIPELINE: det=%d (tracked=%d) | obst=%d | event=%s | speak=%s | GND: %s | "
        "LATENCY: cap=%.1fms inf=%.1fms gnd=%.1fms depth=%.1fms tot=%.1fms submit=%s queue_wait=%.1fms play=%.1fms | "
        "HEALTH: ok=%s depth_ok=%s worker_alive=%s q_depth=%d",
        len(detections), tracked_count,
        len(obstacles),
        event_info,
        speak_result if speak_result is not None else "N/A",
        gnd_summary,
        capture_latency_ms, inference_latency_ms, ground_latency_ms, depth_latency_ms, total_latency_ms,
        f"{submission_latency_ms:.1f}ms" if submission_latency_ms is not None else "N/A",
        queue_wait_ms, playback_dur_ms,
        health_status,
        depth_ok,
        speaker.is_worker_alive if speaker else "N/A",
        speaker.queue_size if speaker else 0,
    )

# Color palette for bounding boxes by proximity (BGR format for OpenCV)
PROXIMITY_COLORS: dict[Proximity, tuple[int, int, int]] = {
    Proximity.NEAR: (0, 0, 255),  # Red
    Proximity.MEDIUM: (0, 165, 255),  # Orange
    Proximity.FAR: (0, 255, 0),  # Green
}

# Text rendering constants
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
FONT_THICKNESS = 1
HUD_FONT_SCALE = 0.6
HUD_FONT_THICKNESS = 2


class Pipeline:
    """End-to-end live narration pipeline.

    Parameters
    ----------
    config : AppConfig, optional
        Top-level configuration.
    detector : Detector, optional
        Custom or mock detector instance.
    speaker : Speaker, optional
        Custom or mock speaker instance.
    """

    def __init__(
        self,
        config: AppConfig | None = None,
        *,
        detector: Detector | None = None,
        speaker: Speaker | None = None,
        health_monitor: HealthMonitor | None = None,
        depth_estimator: DepthEstimator | None = None,
        ground_estimator: GroundRegionEstimator | None = None,
    ) -> None:
        self.config = config or AppConfig()
        self.detector = detector or Detector(self.config.detector)
        self.speaker = speaker or Speaker(self.config.speaker)
        self.health = health_monitor or HealthMonitor(self.config.health)

        # Relative depth estimator (geometric perspective or neural)
        if depth_estimator is not None:
            self.depth_estimator: DepthEstimator | None = depth_estimator
        elif self.config.depth.enabled and self.config.depth.estimator_type == "geometric":
            self.depth_estimator = GeometricDepthEstimator(
                horizon_y=self.config.depth.horizon_y,
                immediate_thresh=self.config.depth.immediate_threshold,
                near_thresh=self.config.depth.near_threshold,
                medium_thresh=self.config.depth.medium_threshold,
                inner_crop=self.config.depth.inner_crop,
            )
        else:
            self.depth_estimator = None

        # Visible ground-region estimator (Phase 2C.1)
        if ground_estimator is not None:
            self.ground_estimator: GroundRegionEstimator = ground_estimator
        else:
            self.ground_estimator = create_ground_estimator(self.config.ground)
        self.last_ground_region: GroundRegion | None = None

        # Event-based narration manager
        self.event_narrator = EventNarrator(
            self.config.events,
            tracking_enabled=self.config.detector.tracking_enabled,
        )

        # Performance monitor
        self.metrics = PerformanceMonitor()

        # Set device name for display
        device_str = self.config.detector.resolve_device()
        if device_str == "0":
            try:
                import torch
                if torch.cuda.is_available():
                    self.metrics.set_device_name(f"CUDA:{torch.cuda.get_device_name(0)}")
                else:
                    self.metrics.set_device_name("CUDA:0")
            except Exception:
                self.metrics.set_device_name("CUDA:0")
        elif device_str == "cpu":
            self.metrics.set_device_name("CPU")
        else:
            self.metrics.set_device_name(device_str)

        self._running = False
        self._previous_zones: dict[int | str | None, HazardZone] = {}

    def process_frame(
        self,
        frame: np.ndarray,
        *,
        capture_latency_ms: float = 0.0,
    ) -> tuple[np.ndarray, list[PrioritizedObstacle], float]:
        """Process a single frame: detect/track, prioritize, trigger events, and annotate.

        Parameters
        ----------
        frame : np.ndarray
            BGR image from the camera.
        capture_latency_ms : float, optional
            Measured frame capture duration in milliseconds.

        Returns
        -------
        annotated_frame : np.ndarray
            Frame with bounding boxes, labels, and stats drawn.
        obstacles : list[PrioritizedObstacle]
            Prioritized obstacle list for this frame.
        latency_ms : float
            Inference latency in milliseconds.
        """
        frame_start = time.perf_counter()

        # Validate input frame
        if frame is None or frame.size == 0:
            cam_alert = self.health.report_frame_capture_failure(error_msg="Empty or None frame")
            if cam_alert and self.config.speaker.enabled:
                self.speaker.speak(cam_alert, urgent=True)
            fallback = frame if frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)
            return fallback, [], 0.0

        # Record capture success
        cam_alert = self.health.report_frame_capture_success(capture_duration_ms=capture_latency_ms)

        h, w = frame.shape[:2]

        # 1. Detection / Tracking
        t0 = time.perf_counter()
        detections = self.detector.detect(frame)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        # Health reporting for detector
        det_alert = None
        if getattr(self.detector, "is_healthy", True):
            det_alert = self.health.report_detector_success(
                len(detections), inference_latency_ms=latency_ms
            )
        else:
            last_err = getattr(self.detector, "last_error", None)
            det_alert = self.health.report_detector_error(last_err)

        # Health reporting for speaker worker
        if hasattr(self.speaker, "is_worker_alive"):
            self.health.report_speaker_status(
                is_worker_alive=self.speaker.is_worker_alive,
                queue_size=getattr(self.speaker, "queue_size", 0),
            )

        # 2. Ground-Region Estimation (Phase 2C.1)
        t_ground0 = time.perf_counter()
        ground_region = None
        if self.ground_estimator is not None:
            try:
                ground_region = self.ground_estimator.estimate_ground_region(frame, detections)
            except Exception as e:
                logger.debug("Ground region estimation failed: %s", e)
                ground_region = None
        self.last_ground_region = ground_region
        ground_latency_ms = (time.perf_counter() - t_ground0) * 1000.0

        # 3. Relative Depth Estimation (Single-pass depth map)
        t_depth0 = time.perf_counter()
        depth_map = None
        depth_ok = False
        if self.depth_estimator is not None and self.config.depth.enabled:
            try:
                depth_map = self.depth_estimator.estimate_depth(frame)
                depth_ok = depth_map is not None and depth_map.size > 0
            except Exception as e:
                logger.debug("Depth estimation failed: %s", e)
                depth_map = None
                depth_ok = False
        depth_latency_ms = (time.perf_counter() - t_depth0) * 1000.0

        # 4. Prioritization
        corridor = self.config.corridor
        obstacles = prioritize_detections(
            detections,
            frame_width=w,
            frame_height=h,
            corridor_x_min=corridor.corridor_x_min,
            corridor_x_max=corridor.corridor_x_max,
            ground_y_min=corridor.ground_y_min,
            path_urgency_bonus=corridor.path_urgency_bonus if corridor.enabled else 0.0,
            depth_map=depth_map,
            depth_estimator=self.depth_estimator,
            previous_zones=self._previous_zones,
            ground_region=ground_region,
        )
        self._previous_zones = {
            (o.track_id if o.track_id is not None else o.label): o.contact_point.zone
            for o in obstacles
            if o.contact_point is not None
        }

        # Debug: Log detections and obstacles
        if self.config.debug_detections:
            narration_logger.info(
                "Frame: detections=%d, obstacles=%d",
                len(detections), len(obstacles)
            )
            for d in detections:
                narration_logger.debug(
                    "  DET: %s conf=%.2f track_id=%s",
                    d.label, d.confidence, d.track_id
                )
            for o in obstacles:
                narration_logger.debug(
                    "  OBST: %s prox=%s dir=%s urgency=%.1f track_id=%s depth=%s",
                    o.label, o.proximity.value, o.direction.value, o.urgency_score, o.track_id, o.relative_depth
                )

        # 4. Event-based Narration
        event = self.event_narrator.update(obstacles, frame_width=w)
        speak_result = None
        submission_latency_ms = None
        if event is not None:
            submission_latency_ms = (time.monotonic() - event.created_at) * 1000.0
            if self.config.debug_detections:
                narration_logger.info(
                    "  EVENT: %s type=%s track_id=%s urgency=%.1f speech='%s' submit_delay=%.1fms",
                    event.label, event.event_type.value, event.track_id, event.urgency_score, event.speech_text, submission_latency_ms
                )
            if self.config.speaker.enabled:
                # Record that this event was offered (prevents re-offering on rejection)
                self.event_narrator.record_offered(event)
                # Check urgency for speech preemption
                from narrator.events import EventType
                is_urgent = (
                    event.event_type == EventType.PROXIMITY_ESCALATION
                    and event.proximity == Proximity.NEAR
                )
                # Only mark event as spoken if speaker actually accepts it
                if is_urgent:
                    speak_result = self.speaker.speak(event.speech_text, urgent=True)
                else:
                    speak_result = self.speaker.speak(event.speech_text)
                if self.config.debug_detections:
                    narration_logger.info(
                        "  SPEAK: accepted=%s text='%s'",
                        speak_result, event.speech_text
                    )
                if speak_result:
                    self.event_narrator.record_spoken(event)
                    if self.config.debug_detections:
                        narration_logger.debug("  RECORD_SPOKEN: event recorded")
            self.health.report_speech_submission(
                speak_result if speak_result is not None else False,
                reason="rejected_or_muted" if not speak_result else None,
            )
        elif det_alert and self.config.speaker.enabled:
            # Genuine obstacle alert not present; announce detector failure or recovery
            is_urgent_alert = (det_alert == "Obstacle detection unavailable.")
            self.speaker.speak(det_alert, urgent=is_urgent_alert)
        elif cam_alert and self.config.speaker.enabled:
            # Genuine obstacle alert not present; announce camera recovery
            self.speaker.speak(cam_alert)
        elif self.config.debug_detections:
            narration_logger.debug("  EVENT: None (no eligible event)")

        # 4. Record Performance Metrics
        total_latency_ms = (time.perf_counter() - frame_start) * 1000.0
        self.metrics.record_frame(
            inference_latency_ms=latency_ms,
            total_latency_ms=total_latency_ms,
            active_tracks=len(obstacles),
            capture_latency_ms=capture_latency_ms,
            submission_latency_ms=submission_latency_ms,
        )

        # Throttled stage-by-stage diagnostic logging
        if self.config.debug_detections:
            _maybe_log_diagnostics(
                detections=detections,
                obstacles=obstacles,
                event=event,
                speak_result=speak_result,
                config=self.config,
                speaker=self.speaker,
                health=self.health,
                submission_latency_ms=submission_latency_ms,
                capture_latency_ms=capture_latency_ms,
                inference_latency_ms=latency_ms,
                depth_latency_ms=depth_latency_ms,
                total_latency_ms=total_latency_ms,
                depth_ok=depth_ok,
                ground_latency_ms=ground_latency_ms,
                ground_region=self.last_ground_region,
            )

        # 5. Render Annotations
        annotated = self._annotate(frame, obstacles, latency_ms)
        return annotated, obstacles, latency_ms

    def _annotate(
        self,
        frame: np.ndarray,
        obstacles: list[PrioritizedObstacle],
        latency_ms: float,
    ) -> np.ndarray:
        """Draw bounding boxes, directional zones, and enhanced stats banner."""
        canvas = frame.copy()
        h, w = canvas.shape[:2]

        # Draw directional zone divider lines (subtle dashed/dotted look)
        left_x = int(w * 0.33)
        right_x = int(w * 0.67)
        cv2.line(canvas, (left_x, 0), (left_x, h), (128, 128, 128), 1, cv2.LINE_AA)
        cv2.line(canvas, (right_x, 0), (right_x, h), (128, 128, 128), 1, cv2.LINE_AA)

        # Draw walking corridor trapezoid if valid ground region is available
        if (
            self.last_ground_region is not None
            and self.last_ground_region.is_valid
            and len(self.last_ground_region.corridor_polygon) >= 4
        ):
            pts = np.int32(self.last_ground_region.corridor_polygon)
            cv2.polylines(canvas, [pts], isClosed=True, color=(200, 180, 0), thickness=1, lineType=cv2.LINE_AA)

        # Draw obstacles
        if self.config.display.show_boxes:
            for obs in obstacles:
                det = obs.detection
                color = PROXIMITY_COLORS.get(obs.proximity, (0, 255, 0))
                pt1 = (int(det.x1), int(det.y1))
                pt2 = (int(det.x2), int(det.y2))
                cv2.rectangle(canvas, pt1, pt2, color, 2)

                if self.config.display.show_labels:
                    # Explicit hazard zone tag: [PATH], [PERI], [ELEV]
                    if obs.contact_point is not None:
                        from narrator.ground_plane import HazardZone
                        if obs.contact_point.zone == HazardZone.WALKING_PATH:
                            zone_tag = " [PATH]"
                        elif obs.contact_point.zone == HazardZone.PERIPHERAL:
                            zone_tag = " [PERI]"
                        else:
                            zone_tag = " [ELEV]"
                    else:
                        zone_tag = ""

                    depth_tag = f", d={obs.relative_depth:.2f}" if obs.relative_depth is not None else ""
                    if obs.track_id is not None:
                        label_text = f"#{obs.track_id} {det.label}{zone_tag} ({obs.proximity.value}, {obs.direction.value}{depth_tag})"
                    else:
                        label_text = f"{det.label}{zone_tag} ({obs.proximity.value}, {obs.direction.value}{depth_tag})"

                    # Draw high-contrast contact point indicator on ground (outer white ring + inner core)
                    if obs.contact_point is not None:
                        cp_x = int(obs.contact_point.x)
                        cp_y = int(obs.contact_point.y)
                        cv2.circle(canvas, (cp_x, cp_y), 6, (255, 255, 255), 2, cv2.LINE_AA)
                        cv2.circle(canvas, (cp_x, cp_y), 3, color, -1, cv2.LINE_AA)

                    # Background rectangle for label readability
                    (label_w, label_h), baseline = cv2.getTextSize(
                        label_text, FONT, FONT_SCALE, FONT_THICKNESS
                    )
                    top_y = max(pt1[1] - label_h - 6, 0)
                    cv2.rectangle(
                        canvas,
                        (pt1[0], top_y),
                        (pt1[0] + label_w + 4, pt1[1]),
                        (20, 20, 20),
                        -1,
                    )
                    cv2.putText(
                        canvas,
                        label_text,
                        (pt1[0] + 2, pt1[1] - 4),
                        FONT,
                        FONT_SCALE,
                        color,
                        FONT_THICKNESS,
                        cv2.LINE_AA,
                    )

        # Draw HUD banner if FPS display enabled
        if self.config.display.show_fps:
            summary = self.metrics.get_summary()

            # Draw semi-transparent top banner
            overlay = canvas.copy()
            cv2.rectangle(overlay, (0, 0), (w, 40), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.65, canvas, 0.35, 0, canvas)

            # Left stats text
            fps_val = summary["fps"]
            inf_val = summary["avg_inference_ms"]
            tot_val = summary["avg_total_ms"]
            hud_text = f"FPS: {fps_val:.1f} | Inf: {inf_val:.0f}ms | Tot: {tot_val:.0f}ms | Tracks: {len(obstacles)}"
            cv2.putText(
                canvas,
                hud_text,
                (10, 25),
                FONT,
                HUD_FONT_SCALE,
                (0, 255, 255),
                HUD_FONT_THICKNESS,
                cv2.LINE_AA,
            )

            # Right badges: Device & Mute status
            right_offset = w - 10
            if not self.config.speaker.enabled:
                mute_text = "[MUTED]"
                (m_w, _), _ = cv2.getTextSize(mute_text, FONT, HUD_FONT_SCALE, HUD_FONT_THICKNESS)
                cv2.putText(
                    canvas,
                    mute_text,
                    (right_offset - m_w, 25),
                    FONT,
                    HUD_FONT_SCALE,
                    (0, 0, 255),  # Red
                    HUD_FONT_THICKNESS,
                    cv2.LINE_AA,
                )
                right_offset -= (m_w + 15)

            # Dynamic Ground status badge (Phase 2C.1)
            if not self.config.ground.enabled or self.ground_estimator is None:
                gnd_mode = "[GND: OFF]"
                gnd_color = (128, 128, 128)
            elif self.last_ground_region is None or not self.last_ground_region.is_valid:
                gnd_mode = "[GND: ERR]"
                gnd_color = (0, 0, 255)  # Red
            elif self.last_ground_region.confidence < 0.50:
                pct = int(self.last_ground_region.confidence * 100)
                gnd_mode = f"[GND: DEG {pct}%]"
                gnd_color = (0, 165, 255)  # Orange / degraded
            else:
                pct = int(self.last_ground_region.confidence * 100)
                gnd_mode = f"[GND: ON {pct}%]"
                gnd_color = (0, 255, 150)  # Green

            (g_w, _), _ = cv2.getTextSize(gnd_mode, FONT, 0.5, 1)
            cv2.putText(
                canvas,
                gnd_mode,
                (right_offset - g_w, 25),
                FONT,
                0.5,
                gnd_color,
                1,
                cv2.LINE_AA,
            )
            right_offset -= (g_w + 15)

            # Depth status badge
            depth_mode = "[DEPTH: ON]" if (self.config.depth.enabled and self.depth_estimator is not None) else "[DEPTH: OFF]"
            (d_w, _), _ = cv2.getTextSize(depth_mode, FONT, 0.5, 1)
            cv2.putText(
                canvas,
                depth_mode,
                (right_offset - d_w, 25),
                FONT,
                0.5,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )
            right_offset -= (d_w + 15)

            # Tracker status badge
            track_mode = "[TRACK: ON]" if self.config.detector.tracking_enabled else "[TRACK: OFF]"
            (t_w, _), _ = cv2.getTextSize(track_mode, FONT, 0.5, 1)
            cv2.putText(
                canvas,
                track_mode,
                (right_offset - t_w, 25),
                FONT,
                0.5,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )
            right_offset -= (t_w + 15)

            # Health status badge if any component is degraded
            if not self.health.is_healthy():
                snapshot = self.health.get_snapshot()
                health_text = f"[WARN: {','.join(snapshot.active_warnings)}]"
                (h_w, _), _ = cv2.getTextSize(health_text, FONT, 0.5, 1)
                cv2.putText(
                    canvas,
                    health_text,
                    (right_offset - h_w, 25),
                    FONT,
                    0.5,
                    (0, 165, 255),  # Orange warning
                    1,
                    cv2.LINE_AA,
                )

        return canvas

    def run(self) -> None:
        """Start the camera loop and process frames until user quits."""
        camera_idx = self.config.camera.index
        logger.info("Opening camera index=%d", camera_idx)
        cap = cv2.VideoCapture(camera_idx)

        if not cap.isOpened():
            cam_alert = self.health.report_camera_open_failed(
                error_msg=f"Failed to open camera with index {camera_idx}"
            )
            if cam_alert and self.config.speaker.enabled:
                self.speaker.speak(cam_alert, urgent=True)
                time.sleep(1.0)
            raise RuntimeError(
                f"Failed to open camera with index {camera_idx}. "
                "Ensure a webcam is connected and not in use by another app."
            )

        # Apply optional camera resolution overrides
        if self.config.camera.frame_width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera.frame_width)
        if self.config.camera.frame_height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera.frame_height)

        self._running = True
        fps_counter = 0
        fps_start = time.perf_counter()
        current_fps = 0.0

        print(f"\n=== Smart Obstacle Narrator ===")
        print(f"Camera index: {camera_idx}")
        print(f"Device: {self.config.detector.resolve_device()}")
        print(f"Speech enabled: {self.config.speaker.enabled}")
        print("Press 'q' in the camera window to quit.\n")

        try:
            while self._running:
                t_cap = time.perf_counter()
                ret, frame = cap.read()
                cap_latency_ms = (time.perf_counter() - t_cap) * 1000.0
                if not ret or frame is None:
                    logger.warning("Failed to grab frame from camera")
                    cam_alert = self.health.report_frame_capture_failure()
                    if cam_alert and self.config.speaker.enabled:
                        self.speaker.speak(cam_alert, urgent=True)
                    time.sleep(0.01)
                    continue

                annotated, obstacles, latency_ms = self.process_frame(
                    frame, capture_latency_ms=cap_latency_ms
                )

                # FPS calculation
                fps_counter += 1
                now = time.perf_counter()
                if (now - fps_start) >= 1.0:
                    current_fps = fps_counter / (now - fps_start)
                    fps_counter = 0
                    fps_start = now

                # Overlay overall FPS
                if self.config.display.show_fps:
                    cv2.putText(
                        annotated,
                        f"FPS: {current_fps:.1f}",
                        (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                cv2.imshow(self.config.display.window_name, annotated)

                # Exit on 'q' or window close
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if cv2.getWindowProperty(self.config.display.window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break

        finally:
            self._running = False
            cap.release()
            cv2.destroyAllWindows()
            self.speaker.stop()
            logger.info("Camera released, pipeline shut down cleanly")

    def stop(self) -> None:
        """Signal the pipeline to stop running."""
        self._running = False
