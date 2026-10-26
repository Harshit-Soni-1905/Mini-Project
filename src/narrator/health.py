"""System health monitoring and trustworthy alert management for Smart Obstacle Narrator.

Tracks operational health across camera, detector, and speaker subsystems.
Distinguishes component failures from normal operational conditions (such as
zero detections on valid frames), and manages concise spoken health transitions
while avoiding alert fatigue and false inferences.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from narrator.config import HealthConfig

logger = logging.getLogger(__name__)


class ComponentStatus(str, Enum):
    """Operational status of a subsystem component."""

    OK = "ok"
    UNAVAILABLE = "unavailable"
    STALE = "stale"
    ERROR = "error"


class HealthCondition(str, Enum):
    """Specific health conditions distinguished by the monitor."""

    CAMERA_OPEN_FAILED = "camera_open_failed"
    FRAME_CAPTURE_FAILED = "frame_capture_failed"
    FRAME_STALE = "frame_stale"
    DETECTOR_EXCEPTION = "detector_exception"
    DETECTOR_ZERO_DETECTIONS = "detector_zero_detections"  # Normal, not a failure
    SPEECH_SUBMISSION_REJECTED = "speech_submission_rejected"
    SPEECH_WORKER_FAILED = "speech_worker_failed"
    PROCESSING_DELAY_EXCEEDED = "processing_delay_exceeded"


@dataclass
class ComponentHealthState:
    """State record for a single subsystem component."""

    status: ComponentStatus = ComponentStatus.OK
    last_transition_time: float = 0.0
    last_spoken_warning_time: float = -float("inf")
    consecutive_failures: int = 0
    total_failures: int = 0
    last_error: str | None = None


@dataclass
class HealthSnapshot:
    """Point-in-time snapshot of system health."""

    camera_status: ComponentStatus
    detector_status: ComponentStatus
    speaker_status: ComponentStatus
    is_healthy: bool
    active_warnings: list[str] = field(default_factory=list)


class HealthMonitor:
    """Monitors system health and produces rate-limited spoken transition warnings.

    Key principles:
    1. Distinguishes confirmed failures from normal conditions (e.g. 0 detections).
    2. Avoids false inferences: 0 detections != path clear, 0 detections != blocked lens.
    3. Speaks only on confirmed transitions into/out of failure states.
    4. Rate-limits repeated warnings to prevent audio loops.
    5. Avoids recursive audio failure announcements over dead TTS.
    6. Ensures genuine obstacle alerts are never suppressed by health alerts.
    """

    WARNING_PHRASES: dict[str, dict[ComponentStatus, str]] = {
        "camera": {
            ComponentStatus.UNAVAILABLE: "Camera unavailable.",
            ComponentStatus.STALE: "Camera feed stale.",
            ComponentStatus.OK: "Camera recovered.",
        },
        "detector": {
            ComponentStatus.UNAVAILABLE: "Obstacle detection unavailable.",
            ComponentStatus.ERROR: "Obstacle detection unavailable.",
            ComponentStatus.OK: "Obstacle detection recovered.",
        },
        # Speaker failures are intentionally NOT spoken to prevent recursive dead-audio loops.
        "speaker": {},
    }

    def __init__(
        self,
        config: HealthConfig | None = None,
        *,
        warning_cooldown: float = 30.0,
        consecutive_failures_to_alert: int = 5,
        stale_threshold_seconds: float = 2.0,
    ) -> None:
        from narrator.config import HealthConfig

        self._config = config or HealthConfig()
        self._warning_cooldown = (
            self._config.warning_cooldown_seconds
            if config is not None
            else warning_cooldown
        )
        self._consecutive_failures_to_alert = (
            self._config.consecutive_capture_failures_to_alert
            if config is not None
            else consecutive_failures_to_alert
        )
        self._stale_threshold = (
            self._config.stale_frame_threshold_seconds
            if config is not None
            else stale_threshold_seconds
        )

        self._camera_state = ComponentHealthState()
        self._detector_state = ComponentHealthState()
        self._speaker_state = ComponentHealthState()

        self._last_valid_frame_time: float | None = None
        self._consecutive_zero_detection_frames: int = 0
        self._total_zero_detection_frames: int = 0

    # ----------------------------------------------------------------------
    # Camera Health Reporting
    # ----------------------------------------------------------------------

    def report_camera_open_failed(
        self,
        *,
        current_time: float | None = None,
        error_msg: str | None = None,
    ) -> str | None:
        """Record that the camera device could not be opened."""
        now = current_time if current_time is not None else time.monotonic()
        self._camera_state.consecutive_failures += 1
        self._camera_state.total_failures += 1
        self._camera_state.last_error = error_msg or "Camera open failed"
        return self._transition_component("camera", self._camera_state, ComponentStatus.UNAVAILABLE, now)

    def report_frame_capture_success(
        self,
        *,
        current_time: float | None = None,
        capture_duration_ms: float = 0.0,
    ) -> str | None:
        """Record a successfully read camera frame.

        Transitions camera back to OK if previously unavailable or stale.
        """
        now = current_time if current_time is not None else time.monotonic()
        self._last_valid_frame_time = now
        self._camera_state.consecutive_failures = 0
        self._camera_state.last_error = None

        # Transition back to OK if previously unavailable or stale
        if self._camera_state.status != ComponentStatus.OK:
            return self._transition_component("camera", self._camera_state, ComponentStatus.OK, now)

        return None

    def check_stale_frame(
        self,
        *,
        current_time: float | None = None,
    ) -> str | None:
        """Check if frame interval has exceeded stale threshold."""
        now = current_time if current_time is not None else time.monotonic()
        if (
            self._last_valid_frame_time is not None
            and (now - self._last_valid_frame_time) > self._stale_threshold
        ):
            logger.warning(
                "Camera frame interval (%.2fs) exceeded stale threshold (%.2fs)",
                now - self._last_valid_frame_time,
                self._stale_threshold,
            )
            return self._transition_component(
                "camera", self._camera_state, ComponentStatus.STALE, now
            )
        return None

    def report_frame_capture_failure(
        self,
        *,
        current_time: float | None = None,
        error_msg: str | None = None,
    ) -> str | None:
        """Record a failed frame read (e.g. ret=False from cv2.VideoCapture).

        Only triggers an alert after consecutive failures exceed the threshold,
        preventing spurious warnings on transient single-frame glitches.
        """
        now = current_time if current_time is not None else time.monotonic()
        self._camera_state.consecutive_failures += 1
        self._camera_state.total_failures += 1
        self._camera_state.last_error = error_msg or "Frame read returned False"

        # Check if frame is also stale based on elapsed time since last valid frame
        stale_alert = self.check_stale_frame(current_time=now)
        if stale_alert:
            return stale_alert

        if self._camera_state.consecutive_failures >= self._consecutive_failures_to_alert:
            return self._transition_component(
                "camera", self._camera_state, ComponentStatus.UNAVAILABLE, now
            )

        return None

    # ----------------------------------------------------------------------
    # Detector Health Reporting
    # ----------------------------------------------------------------------

    def report_detector_error(
        self,
        exc: Exception | str | None = None,
        *,
        current_time: float | None = None,
    ) -> str | None:
        """Record an inference exception or unavailability."""
        now = current_time if current_time is not None else time.monotonic()
        self._detector_state.consecutive_failures += 1
        self._detector_state.total_failures += 1
        self._detector_state.last_error = str(exc) if exc is not None else "Detector exception"
        logger.error("Detector inference error reported: %s", self._detector_state.last_error)
        return self._transition_component("detector", self._detector_state, ComponentStatus.UNAVAILABLE, now)

    def report_detector_success(
        self,
        detection_count: int,
        *,
        inference_latency_ms: float = 0.0,
        current_time: float | None = None,
    ) -> str | None:
        """Record successful detector inference.

        Note: detection_count == 0 is completely normal and valid. It is NOT
        treated as an error or obstruction. Zero detections do NOT imply
        that the path is clear or that the camera is obstructed.
        """
        now = current_time if current_time is not None else time.monotonic()
        self._detector_state.consecutive_failures = 0
        self._detector_state.last_error = None

        if detection_count == 0:
            self._consecutive_zero_detection_frames += 1
            self._total_zero_detection_frames += 1
        else:
            self._consecutive_zero_detection_frames = 0

        # Transition back to OK if previously in error/unavailable
        if self._detector_state.status != ComponentStatus.OK:
            return self._transition_component("detector", self._detector_state, ComponentStatus.OK, now)

        return None

    # ----------------------------------------------------------------------
    # Speaker Health Reporting
    # ----------------------------------------------------------------------

    def report_speaker_status(
        self,
        *,
        is_worker_alive: bool,
        queue_size: int,
        current_time: float | None = None,
    ) -> None:
        """Track speaker worker health.

        Does NOT emit a spoken warning if speaker is dead, preventing
        recursive failure attempts over dead audio.
        """
        now = current_time if current_time is not None else time.monotonic()
        if not is_worker_alive:
            if self._speaker_state.status != ComponentStatus.UNAVAILABLE:
                logger.error("TTS speaker worker thread is not alive")
                self._speaker_state.status = ComponentStatus.UNAVAILABLE
                self._speaker_state.last_transition_time = now
                self._speaker_state.consecutive_failures += 1
        else:
            if self._speaker_state.status != ComponentStatus.OK:
                logger.info("TTS speaker worker thread is healthy")
                self._speaker_state.status = ComponentStatus.OK
                self._speaker_state.last_transition_time = now
                self._speaker_state.consecutive_failures = 0

    def report_speech_submission(
        self,
        accepted: bool,
        *,
        reason: str | None = None,
        current_time: float | None = None,
    ) -> None:
        """Track speech submission acceptances and rejections for diagnostics."""
        if not accepted:
            self._speaker_state.total_failures += 1
            logger.debug("Speech submission rejected (reason=%s)", reason)

    # ----------------------------------------------------------------------
    # Internal State Transitions & Rate Limiting
    # ----------------------------------------------------------------------

    def _transition_component(
        self,
        component: str,
        state: ComponentHealthState,
        new_status: ComponentStatus,
        now: float,
    ) -> str | None:
        """Transition a component status and return a spoken phrase if eligible."""
        old_status = state.status
        if new_status == old_status:
            return None

        logger.info(
            "Health transition for [%s]: %s -> %s (time=%.2fs)",
            component,
            old_status.value,
            new_status.value,
            now,
        )
        state.status = new_status
        state.last_transition_time = now

        # Only announce if there is a defined phrase
        phrases = self.WARNING_PHRASES.get(component, {})
        phrase = phrases.get(new_status)
        if phrase is None:
            return None

        # Recovery transitions (transitioning back to OK) are announced when service is restored.
        if new_status == ComponentStatus.OK:
            logger.info("Spoken health recovery triggered: %r", phrase)
            return phrase

        # Failure transitions (entering UNAVAILABLE / STALE / ERROR) are rate-limited
        # to avoid repetitive warning loops if failures flicker.
        if (now - state.last_spoken_warning_time) >= self._warning_cooldown:
            state.last_spoken_warning_time = now
            logger.info("Spoken health alert triggered: %r", phrase)
            return phrase
        else:
            logger.debug(
                "Spoken health alert suppressed by cooldown: %r (time=%.2fs)",
                phrase,
                now,
            )
            return None

    # ----------------------------------------------------------------------
    # Query Methods
    # ----------------------------------------------------------------------

    def get_status(self, component: str) -> ComponentStatus:
        """Return the current operational status of a component."""
        if component == "camera":
            return self._camera_state.status
        elif component == "detector":
            return self._detector_state.status
        elif component == "speaker":
            return self._speaker_state.status
        return ComponentStatus.ERROR

    def is_healthy(self) -> bool:
        """Return True if all components are in the OK status."""
        return (
            self._camera_state.status == ComponentStatus.OK
            and self._detector_state.status == ComponentStatus.OK
            and self._speaker_state.status == ComponentStatus.OK
        )

    def is_perception_available(self) -> bool:
        """Return True if camera and detector are operational."""
        return (
            self._camera_state.status == ComponentStatus.OK
            and self._detector_state.status == ComponentStatus.OK
        )

    @property
    def consecutive_zero_detections(self) -> int:
        """Number of consecutive frames with zero detections."""
        return self._consecutive_zero_detection_frames

    def get_snapshot(self) -> HealthSnapshot:
        """Return a complete snapshot of current health states."""
        active_warnings = []
        if self._camera_state.status != ComponentStatus.OK:
            active_warnings.append(f"camera:{self._camera_state.status.value}")
        if self._detector_state.status != ComponentStatus.OK:
            active_warnings.append(f"detector:{self._detector_state.status.value}")
        if self._speaker_state.status != ComponentStatus.OK:
            active_warnings.append(f"speaker:{self._speaker_state.status.value}")

        return HealthSnapshot(
            camera_status=self._camera_state.status,
            detector_status=self._detector_state.status,
            speaker_status=self._speaker_state.status,
            is_healthy=self.is_healthy(),
            active_warnings=active_warnings,
        )

    def reset(self) -> None:
        """Reset all health states to default OK."""
        self._camera_state = ComponentHealthState()
        self._detector_state = ComponentHealthState()
        self._speaker_state = ComponentHealthState()
        self._last_valid_frame_time = None
        self._consecutive_zero_detection_frames = 0
        self._total_zero_detection_frames = 0
