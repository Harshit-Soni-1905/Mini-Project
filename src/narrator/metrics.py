"""Rolling performance monitor for the narration pipeline.

Provides lightweight, rolling-window metrics (FPS, inference latency,
total frame latency, tracked object count) without expensive GPU
synchronization on every frame.
"""

from __future__ import annotations

import time
from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


class PerformanceMonitor:
    """Rolling-window performance metrics for the live pipeline.

    Tracks:
    - Rolling FPS (frames per second) over the last N frames.
    - Average inference latency (ms) over the last N frames.
    - Average total frame latency (ms) over the last N frames.
    - Active track count per frame.

    Uses a fixed-size deque for constant memory and O(1) operations.
    Excludes warm-up frames from steady-state averages.
    """

    def __init__(
        self,
        window_size: int = 30,
        warmup_frames: int = 3,
    ) -> None:
        """
        Args:
            window_size: Number of frames to keep in rolling window.
            warmup_frames: Number of initial frames to exclude from averages.
        """
        self._window_size = window_size
        self._warmup_frames = warmup_frames
        self._frame_count = 0

        # Rolling windows (maxlen = window_size)
        self._inference_latencies: deque[float] = deque(maxlen=window_size)
        self._total_latencies: deque[float] = deque(maxlen=window_size)
        self._capture_latencies: deque[float] = deque(maxlen=window_size)
        self._submission_latencies: deque[float] = deque(maxlen=window_size)
        self._frame_intervals: deque[float] = deque(maxlen=window_size)
        self._track_counts: deque[int] = deque(maxlen=window_size)

        self._last_frame_time: float | None = None
        self._device_name: str = "Unknown"

    def set_device_name(self, device_name: str) -> None:
        """Set the inference device name (e.g., 'CUDA:0', 'CPU')."""
        self._device_name = device_name

    def record_frame(
        self,
        inference_latency_ms: float,
        total_latency_ms: float,
        active_tracks: int = 0,
        capture_latency_ms: float = 0.0,
        submission_latency_ms: float | None = None,
    ) -> None:
        """Record metrics for a single processed frame."""
        self._frame_count += 1
        now = time.monotonic()

        # Record latencies
        self._inference_latencies.append(inference_latency_ms)
        self._total_latencies.append(total_latency_ms)
        self._capture_latencies.append(capture_latency_ms)
        if submission_latency_ms is not None:
            self._submission_latencies.append(submission_latency_ms)
        self._track_counts.append(active_tracks)

        # Record frame interval for FPS calculation
        if self._last_frame_time is not None:
            interval = now - self._last_frame_time
            self._frame_intervals.append(interval)
        self._last_frame_time = now

    @property
    def fps(self) -> float:
        """Rolling average FPS over the window (excluding warmup)."""
        if len(self._frame_intervals) < 2:
            return 0.0
        valid_intervals = list(self._frame_intervals)[self._warmup_frames - 1 :]
        if len(valid_intervals) < 2:
            return 0.0
        avg_interval = sum(valid_intervals) / len(valid_intervals)
        return 1.0 / avg_interval if avg_interval > 0 else 0.0

    @property
    def avg_inference_ms(self) -> float:
        """Average inference latency in milliseconds (excluding warmup)."""
        if not self._inference_latencies:
            return 0.0
        valid = list(self._inference_latencies)[self._warmup_frames - 1 :]
        if not valid:
            return 0.0
        return sum(valid) / len(valid)

    @property
    def avg_total_ms(self) -> float:
        """Average total frame latency in milliseconds (excluding warmup)."""
        if not self._total_latencies:
            return 0.0
        valid = list(self._total_latencies)[self._warmup_frames - 1 :]
        if not valid:
            return 0.0
        return sum(valid) / len(valid)

    @property
    def avg_capture_ms(self) -> float:
        """Average capture latency in milliseconds (excluding warmup)."""
        if not self._capture_latencies:
            return 0.0
        valid = list(self._capture_latencies)[self._warmup_frames - 1 :]
        if not valid:
            return 0.0
        return sum(valid) / len(valid)

    @property
    def avg_submission_ms(self) -> float:
        """Average event creation to speech submission latency in milliseconds."""
        if not self._submission_latencies:
            return 0.0
        return sum(self._submission_latencies) / len(self._submission_latencies)

    @property
    def active_tracks(self) -> int:
        """Most recent tracked object count."""
        if not self._track_counts:
            return 0
        return self._track_counts[-1]

    @property
    def device_name(self) -> str:
        """Inference device name."""
        return self._device_name

    @property
    def is_warmed_up(self) -> bool:
        """Whether enough frames have been processed for steady-state metrics."""
        return self._frame_count >= self._warmup_frames

    def get_summary(self) -> dict[str, float | str | int]:
        """Return a snapshot of current metrics for display."""
        return {
            "fps": round(self.fps, 1),
            "avg_inference_ms": round(self.avg_inference_ms, 1),
            "avg_total_ms": round(self.avg_total_ms, 1),
            "avg_capture_ms": round(self.avg_capture_ms, 1),
            "avg_submission_ms": round(self.avg_submission_ms, 1),
            "active_tracks": self.active_tracks,
            "device": self._device_name,
            "warmed_up": self.is_warmed_up,
        }

    def reset(self) -> None:
        """Clear all recorded metrics."""
        self._frame_count = 0
        self._inference_latencies.clear()
        self._total_latencies.clear()
        self._capture_latencies.clear()
        self._submission_latencies.clear()
        self._frame_intervals.clear()
        self._track_counts.clear()
        self._last_frame_time = None