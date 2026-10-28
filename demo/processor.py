"""WebRTC video frame processor for Smart Obstacle Narrator.

Connects incoming browser webcam frames to an isolated Pipeline instance,
running YOLO detection, tracking, ground reasoning, depth estimation, and
event narration. Returns annotated frames to the WebRTC video stream.
"""

from __future__ import annotations

import logging
import time

import av
import cv2
import numpy as np
from streamlit_webrtc import VideoProcessorBase

from demo.web_speaker import WebSpeechBridge
from narrator.config import AppConfig
from narrator.pipeline import Pipeline
from narrator.prioritizer import PrioritizedObstacle

logger = logging.getLogger(__name__)


class NarratorVideoProcessor(VideoProcessorBase):
    """Session-isolated WebRTC video frame processor.

    Every browser connection instantiates its own NarratorVideoProcessor,
    ensuring that ByteTrack object IDs, ground hysteresis zones, and event
    cooldowns remain strictly isolated per visitor.
    """

    def __init__(self) -> None:
        # Load configuration (supports NARRATOR_MODEL env override)
        self.config = AppConfig.from_env()

        # Enable speaker in config so Pipeline forwards events to speaker.speak(),
        # but inject WebSpeechBridge instead of pyttsx3.
        self.config.speaker.enabled = True

        # In-memory speech bridge for this session
        self.speech_bridge = WebSpeechBridge()

        # Isolated Pipeline instance with dedicated Detector and EventNarrator
        self.pipeline = Pipeline(self.config, speaker=self.speech_bridge)

        # Thread-safe telemetry for the Streamlit UI
        self.last_latency_ms: float = 0.0
        self.last_total_ms: float = 0.0
        self.last_obstacles: list[PrioritizedObstacle] = []
        self.frame_count: int = 0
        self._last_process_timestamp: float = 0.0
        self._min_frame_interval: float = 0.033  # Cap processing at ~30 FPS

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """Process a single WebRTC frame from the browser webcam.

        Parameters
        ----------
        frame : av.VideoFrame
            Incoming video frame from the client's browser.

        Returns
        -------
        av.VideoFrame
            Annotated video frame returned to the browser canvas.
        """
        t_start = time.perf_counter()

        # Convert PyAV frame to BGR NumPy array (in-memory only, never saved to disk)
        try:
            img = frame.to_ndarray(format="bgr24")
        except Exception as exc:
            logger.warning("Failed to decode incoming WebRTC video frame: %s", exc)
            return frame

        if img is None or img.size == 0:
            return frame

        # Enforce maximum resolution (640x480) to conserve container CPU and memory
        h, w = img.shape[:2]
        if w > 640 or h > 480:
            scale = min(640.0 / w, 480.0 / h)
            new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Process frame through the session-isolated pipeline
        try:
            annotated, obstacles, latency_ms = self.pipeline.process_frame(img)
            self.last_latency_ms = latency_ms
            self.last_obstacles = obstacles
            self.frame_count += 1
        except Exception as exc:
            logger.exception("Error processing frame in web pipeline: %s", exc)
            annotated = img

        self.last_total_ms = (time.perf_counter() - t_start) * 1000.0

        # Return annotated frame back to the browser WebRTC stream
        try:
            return av.VideoFrame.from_ndarray(annotated, format="bgr24")
        except Exception as exc:
            logger.warning("Failed to encode outgoing WebRTC video frame: %s", exc)
            return frame

    def on_ended(self) -> None:
        """Teardown handler called when the WebRTC connection closes."""
        logger.info("WebRTC session ended. Releasing pipeline and speech resources.")
        if hasattr(self, "pipeline") and self.pipeline is not None:
            self.pipeline.stop()
        if hasattr(self, "speech_bridge") and self.speech_bridge is not None:
            self.speech_bridge.stop()
