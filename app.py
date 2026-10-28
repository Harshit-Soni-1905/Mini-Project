"""Smart Obstacle Narrator — Public Web Demo.

Streamlit Community Cloud web application providing browser webcam obstacle detection,
ground-aware walking corridor reasoning, and live spoken alerts via WebRTC and the
Web Speech API.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

# Ensure 'src/' is available in sys.path when deployed without 'pip install -e .'
_REPO_ROOT = Path(__file__).resolve().parent
_SRC_PATH = _REPO_ROOT / "src"
if str(_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(_SRC_PATH))

import streamlit as st
from streamlit_webrtc import (
    RTCConfiguration,
    WebRtcMode,
    webrtc_streamer,
)

from demo.processor import NarratorVideoProcessor
from narrator.config import DEFAULT_MODEL

logger = logging.getLogger(__name__)

# Configure page settings
st.set_page_config(
    page_title="Smart Obstacle Narrator — Live Assistive Vision Demo",
    page_icon="🦯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Standard public STUN servers for WebRTC NAT traversal
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


@st.cache_resource(show_spinner=False)
def prewarm_yolo_model(model_name: str) -> bool:
    """Pre-warm/download YOLO weights once at startup to prevent WebRTC handshake timeouts."""
    try:
        from ultralytics import YOLO

        YOLO(model_name)
        return True
    except Exception as exc:
        logger.warning("Optional model pre-warm for '%s' failed: %s", model_name, exc)
        return False


def main() -> None:
    st.title("🦯 Smart Obstacle Narrator")
    st.caption("Live Assistive Spatial Vision & Hazard Prioritization Demo")

    # Prominent safety and privacy disclaimers
    st.info(
        "🔒 **Privacy Notice:** Camera frames are processed strictly in-memory (RAM) and immediately discarded. "
        "No images, video, or audio are ever saved, stored, or logged.\n\n"
        "⚠️ **Research Prototype Disclaimer:** This demo is an academic prototype evaluating monocular spatial awareness. "
        "It does not measure metric distance and is **NOT** certified for real-world navigation or blind obstacle avoidance."
    )

    # Sidebar system configuration & status
    active_model = os.environ.get("NARRATOR_MODEL", DEFAULT_MODEL)
    prewarm_yolo_model(active_model)

    with st.sidebar:
        st.header("System Architecture")
        st.markdown(f"**Detector Model:** `{active_model}`")
        st.markdown("**Tracking:** ByteTrack (Session Isolated)")
        st.markdown("**Hazard Reasoning:** Perspective Trapezoid Corridor")
        st.markdown("**Audio Engine:** Client-Side Web Speech API")

        st.markdown("---")
        st.subheader("Bounding Box Legend")
        st.markdown("🔴 **NEAR** — Immediate footstep / critical collision hazard")
        st.markdown("🟠 **MEDIUM** — Approaching obstacle")
        st.markdown("🟢 **FAR** — Background situational awareness")
        st.markdown("🟡 **Trapezoid** — Projected walking path corridor")

        st.markdown("---")
        st.subheader("Speech & Voice Settings")
        enable_audio = st.toggle("Enable Voice Alerts (Browser Audio)", value=True)
        speech_rate = st.slider("Speech Rate", min_value=0.8, max_value=1.5, value=1.1, step=0.1)

        # Test Voice button: Works immediately with or without the camera active
        if st.button("🔊 Test Voice"):
            if enable_audio:
                test_phrase_js = json.dumps("Smart Obstacle Narrator audio is working")
                st.html(
                    f"""
                    <script>
                    (() => {{
                        try {{
                            if ('speechSynthesis' in window) {{
                                if (window.speechSynthesis.paused) {{
                                    window.speechSynthesis.resume();
                                }}
                                const utterance = new SpeechSynthesisUtterance({test_phrase_js});
                                utterance.rate = {speech_rate};
                                utterance.lang = "en-US";
                                window._activeUtterance = utterance;
                                utterance.onstart = () => console.log("Test voice started");
                                utterance.onend = () => {{
                                    console.log("Test voice completed");
                                    window._activeUtterance = null;
                                }};
                                utterance.onerror = (e) => console.error("Test voice error:", e.error);
                                window.speechSynthesis.speak(utterance);
                            }} else {{
                                console.warn("SpeechSynthesis not available in this browser");
                            }}
                        }} catch (e) {{
                            console.warn("Test voice exception:", e);
                        }}
                    }})();
                    </script>
                    """,
                    unsafe_allow_javascript=True,
                )
                st.toast("🔊 Test voice triggered — check browser console if silent", icon="🔊")
            else:
                st.warning("Voice alerts are disabled. Enable the toggle above to test.")

    # Main layout: Video stream column and live telemetry column
    col_video, col_telemetry = st.columns([3, 2])

    with col_video:
        st.subheader("Live Camera Feed")
        webrtc_ctx = webrtc_streamer(
            key="smart-obstacle-narrator",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=NarratorVideoProcessor,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={
                "video": {"width": {"ideal": 640}, "height": {"ideal": 480}},
                "audio": False,
            },
            async_processing=True,
        )

        if not webrtc_ctx.state.playing:
            st.info("👆 Click **START** above and grant webcam access to start the demo.")

    with col_telemetry:
        st.subheader("Live Narration & Telemetry")
        render_telemetry_panel(webrtc_ctx, enable_audio=enable_audio, speech_rate=speech_rate)


@st.fragment(run_every=0.5)
def render_telemetry_panel(webrtc_ctx, enable_audio: bool = True, speech_rate: float = 1.1) -> None:
    """Render auto-refreshing narration captions, speech triggers, and obstacle telemetry."""
    if not webrtc_ctx.state.playing or webrtc_ctx.video_processor is None:
        st.write("Waiting for camera stream to start...")
        return

    processor = webrtc_ctx.video_processor

    # Drain new speech alerts from the session-isolated bridge
    new_alerts = processor.speech_bridge.pop_new_alerts()
    latest_alert = processor.speech_bridge.get_latest_alert()

    # Client-side audio speech dispatch via Web Speech API
    if enable_audio and new_alerts:
        # Trigger speech in the visitor's browser for newly emitted alerts
        # Capability check and execution happen entirely in JavaScript using safe JSON encoding
        alert_json = json.dumps(new_alerts[-1])
        st.html(
            f"""
            <script>
            (() => {{
                try {{
                    if ('speechSynthesis' in window) {{
                        if (window.speechSynthesis.paused) {{
                            window.speechSynthesis.resume();
                        }}
                        const utterance = new SpeechSynthesisUtterance({alert_json});
                        utterance.rate = {speech_rate};
                        utterance.lang = "en-US";
                        window._activeUtterance = utterance;
                        utterance.onend = () => {{
                            window._activeUtterance = null;
                        }};
                        utterance.onerror = (e) => console.error("Alert speech error:", e.error);
                        window.speechSynthesis.speak(utterance);
                    }} else {{
                        console.warn("SpeechSynthesis not available in this browser");
                    }}
                }} catch (e) {{
                    console.warn("SpeechSynthesis error:", e);
                }}
            }})();
            </script>
            """,
            unsafe_allow_javascript=True,
        )

    # Display prominent live caption badge
    if latest_alert:
        st.success(f"🗣️ **Spoken Alert:** {latest_alert}")
    else:
        st.caption("Listening for obstacle events...")

    # Audio engine status indicator
    if not enable_audio:
        st.warning("🔇 **Voice alerts disabled** — Enable the toggle in the sidebar to hear spoken alerts.")
    else:
        st.caption("🔊 Voice alerts enabled — Browser will speak new alerts via Web Speech API. "
                   "If silent, click **Test Voice** or check browser permissions.")

    # Performance metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Inference", f"{processor.last_latency_ms:.1f} ms")
    m2.metric("Total Latency", f"{processor.last_total_ms:.1f} ms")
    m3.metric("Frames Processed", f"{processor.frame_count}")

    # Active Obstacles Table
    st.markdown("#### Detected Obstacles in Scene")
    obstacles = getattr(processor, "last_obstacles", [])
    if obstacles:
        table_rows = []
        for o in obstacles[:5]:  # Top 5 prioritized
            zone_tag = "Walking Path" if o.is_in_walking_path else "Peripheral"
            table_rows.append(
                {
                    "Object": o.label,
                    "Zone": zone_tag,
                    "Proximity": o.proximity.value.upper(),
                    "Direction": o.direction.value.upper(),
                    "Urgency Score": f"{o.urgency_score:.1f}",
                }
            )
        st.dataframe(table_rows, use_container_width=True, hide_index=True)
    else:
        st.caption("No obstacles currently detected in frame.")

    # Recent Narration History
    history = processor.speech_bridge.get_history()
    if history:
        with st.expander("Recent Alert History", expanded=False):
            for text, ts, urgent in reversed(history[-8:]):
                tag = "🚨 URGENT: " if urgent else ""
                st.text(f"• {tag}{text}")


if __name__ == "__main__":
    main()
