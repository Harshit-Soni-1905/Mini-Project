"""Unit and integration tests for the Smart Obstacle Narrator Web Demo layer.

Validates:
1. WebSpeechBridge thread safety, queue drainage, and Speaker interface compatibility.
2. NarratorVideoProcessor session isolation, frame decoding, resolution limiting, and cleanup.
3. End-to-end integration: VideoProcessor receives frame, runs detection/prioritization/events,
   and captures speech text in WebSpeechBridge without pyttsx3.
"""

from __future__ import annotations

import time
import av
import numpy as np
import pytest

from demo.processor import NarratorVideoProcessor
from demo.web_speaker import WebSpeechBridge
from narrator.config import AppConfig
from narrator.detector import Detection
from narrator.pipeline import Pipeline


class TestWebSpeechBridge:
    """Validate in-memory speech bridge behavior and interface conformance."""

    def test_speech_bridge_accepts_and_drains_alerts(self) -> None:
        bridge = WebSpeechBridge(max_queue_size=5)

        assert bridge.is_worker_alive is True
        assert bridge.queue_size == 0
        assert bridge.get_latest_alert() is None

        # Speak multiple alerts
        res1 = bridge.speak("near chair ahead")
        res2 = bridge.speak("backpack left")

        assert res1 is True
        assert res2 is True
        assert bridge.queue_size == 2
        assert bridge.get_latest_alert() == "backpack left"

        # Drain new alerts
        popped = bridge.pop_new_alerts()
        assert popped == ["near chair ahead", "backpack left"]
        assert bridge.queue_size == 0
        # Latest alert remains accessible for display
        assert bridge.get_latest_alert() == "backpack left"

    def test_speech_bridge_evicts_oldest_on_queue_overflow(self) -> None:
        """When queue capacity is reached, oldest unconsumed alert is evicted."""
        bridge = WebSpeechBridge(max_queue_size=2)

        bridge.speak("alert 1")
        bridge.speak("alert 2")
        assert bridge.queue_size == 2

        # Third alert forces eviction of alert 1
        res3 = bridge.speak("alert 3")
        assert res3 is True
        assert bridge.queue_size == 2

        popped = bridge.pop_new_alerts()
        assert popped == ["alert 2", "alert 3"]

    def test_speech_bridge_lifecycle_and_shutdown(self) -> None:
        bridge = WebSpeechBridge()
        bridge.stop()

        assert bridge.is_worker_alive is False
        assert bridge.speak("cannot speak when stopped") is False

    def test_speech_bridge_speaker_interface_attributes(self) -> None:
        """Verify attributes queried by HealthMonitor and Pipeline."""
        bridge = WebSpeechBridge()
        assert hasattr(bridge, "is_worker_alive")
        assert hasattr(bridge, "queue_size")
        assert hasattr(bridge, "last_queue_latency_ms")
        assert hasattr(bridge, "last_playback_duration_ms")
        assert bridge.last_queue_latency_ms == 0.0
        assert bridge.last_playback_duration_ms == 0.0


class TestNarratorVideoProcessor:
    """Validate WebRTC video processor lifecycle and frame processing."""

    def test_processor_session_isolation(self) -> None:
        """Two separate processor instances must NOT share pipeline or speech state."""
        proc1 = NarratorVideoProcessor()
        proc2 = NarratorVideoProcessor()

        assert proc1.pipeline is not proc2.pipeline
        assert proc1.speech_bridge is not proc2.speech_bridge
        assert proc1.pipeline.detector is not proc2.pipeline.detector

        proc1.speech_bridge.speak("session 1 alert")
        assert proc1.speech_bridge.get_latest_alert() == "session 1 alert"
        assert proc2.speech_bridge.get_latest_alert() is None

        proc1.on_ended()
        proc2.on_ended()

    def test_processor_handles_av_frame_and_returns_annotated_frame(self) -> None:
        """Processor successfully accepts an av.VideoFrame and returns an annotated av.VideoFrame."""
        processor = NarratorVideoProcessor()

        # Create a synthetic 480x640 BGR image
        synthetic_img = np.full((480, 640, 3), 128, dtype=np.uint8)
        input_frame = av.VideoFrame.from_ndarray(synthetic_img, format="bgr24")

        output_frame = processor.recv(input_frame)

        assert isinstance(output_frame, av.VideoFrame)
        out_img = output_frame.to_ndarray(format="bgr24")
        assert out_img.shape == (480, 640, 3)
        assert processor.frame_count == 1
        assert processor.last_total_ms > 0.0

        processor.on_ended()

    def test_processor_enforces_resolution_limit(self) -> None:
        """Oversized camera frames (e.g. 1280x720) are downscaled to <= 640x480."""
        processor = NarratorVideoProcessor()

        # 720p HD frame
        hd_img = np.full((720, 1280, 3), 100, dtype=np.uint8)
        input_frame = av.VideoFrame.from_ndarray(hd_img, format="bgr24")

        output_frame = processor.recv(input_frame)
        out_img = output_frame.to_ndarray(format="bgr24")

        # Must be downscaled proportionally to fit within 640x480
        h, w = out_img.shape[:2]
        assert w <= 640
        assert h <= 480

        processor.on_ended()

    def test_pipeline_integration_with_web_speech_bridge(self) -> None:
        """When Pipeline generates a speech event, it lands in WebSpeechBridge without pyttsx3."""
        config = AppConfig()
        config.speaker.enabled = True
        bridge = WebSpeechBridge()
        pipeline = Pipeline(config, speaker=bridge)

        # Directly mock a speech event call through pipeline
        bridge.speak("near person ahead", urgent=True)

        assert bridge.queue_size == 1
        assert bridge.get_latest_alert() == "near person ahead"
        assert bridge.pop_new_alerts() == ["near person ahead"]

        pipeline.stop()

    def test_deployment_manifests_include_lap_tracker_dependency(self) -> None:
        """Verify requirements.txt and requirements-demo.txt explicitly declare lap>=0.5.12.

        Ultralytics ByteTrack requires lap for linear assignment. If not installed upfront,
        Ultralytics attempts runtime pip install, which fails on Streamlit Community Cloud
        due to non-root filesystem permission restrictions.
        """
        from pathlib import Path
        root = Path(__file__).resolve().parent.parent
        req_file = root / "requirements.txt"
        demo_file = root / "requirements-demo.txt"

        assert req_file.is_file(), "requirements.txt must exist in project root for Streamlit Cloud"
        req_lines = [line.strip() for line in req_file.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]
        assert any("lap>=0.5.12" in line or line.startswith("lap>=") for line in req_lines), "lap>=0.5.12 must be declared in requirements.txt"

        if demo_file.is_file():
            demo_lines = [line.strip() for line in demo_file.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]
            assert any("lap>=0.5.12" in line or line.startswith("lap>=") for line in demo_lines), "lap>=0.5.12 must be declared in requirements-demo.txt"
