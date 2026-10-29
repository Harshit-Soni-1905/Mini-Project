# Smart Obstacle Narrator

An assistive computer vision prototype that detects obstacles in a live camera feed, estimates approximate distance and direction using monocular cues, prioritizes warnings by urgency, and announces concise messages via text-to-speech.

> ⚠️ **Research Prototype — Not a Safety Device**
>
> This software is an **experimental prototype** for academic and personal exploration. It is **not** a substitute for a white cane, guide dog, or professional orientation-and-mobility training. Object detection can miss obstacles—especially transparent surfaces, thin objects, and ground-level hazards. Monocular distance estimation is inherently imprecise. **Never rely on this system as a primary navigation aid.**

---

## Features

| Capability | Implementation |
|------------|----------------|
| **Object Detection** | YOLO11 (Ultralytics) — person, vehicle, and everyday object classes |
| **Persistent Tracking** | ByteTrack multi-object tracking with track ID continuity |
| **Spatial Reasoning** | Perspective-aware walking corridor + relative depth from bbox geometry |
| **Hazard Prioritization** | Urgency scoring combining proximity, direction, class, and path intersection |
| **Event-Driven Narration** | New object, proximity escalation, urgency escalation, and periodic reminders with per-track and global cooldowns |
| **Live Spoken Alerts** | Desktop: native SAPI5 (Windows) via pyttsx3 · Web: client-side Web Speech API |
| **WebRTC Streaming Demo** | Browser webcam + serverless inference with session-isolated pipelines |
| **Performance Monitoring** | Rolling-window FPS, inference latency, total latency, and active track count |

---

## Live Demo

**Streamlit Community Cloud Web Demo:** <https://smart-obstacle-narrator-1.streamlit.app/>

> The web demo runs entirely in the browser via WebRTC. Camera frames never leave the visitor's device; obstacle detection and tracking execute on the server (Streamlit Cloud free tier CPU), and only annotated frames + narration text are streamed back. No images, video, or audio are stored or logged.

---

## Quick Start (Local Development)

### Prerequisites
- Python 3.12+ (tested with 3.14.7)
- NVIDIA GPU with CUDA support recommended (tested on RTX 3050 Laptop, 4 GB VRAM)
- Windows 10/11 for native SAPI5 voices; Linux/macOS supported with fallback TTS

### Installation

```bash
# 1. Clone and enter the repository
git clone https://github.com/Harshit-Soni-1905/Mini-Project.git
cd Mini-Project

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

# 3. Install PyTorch with CUDA support (see docs/environment.md for alternatives)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132

# 4. Install the project in editable mode with dev dependencies
pip install -e ".[dev]"
```

### Verify the Setup

```bash
# Run the full test suite (270 tests)
pytest

# Run the baseline detection test on a sample image
python tests/test_baseline_detection.py
```

---

## Usage

### Desktop CLI (Local Camera + Native TTS)

```bash
# Run with default model (yolo11s.pt) on CUDA if available
python -m narrator

# Override model via environment variable
NARRATOR_MODEL=yolo11n.pt python -m narrator
```

| Key | Action |
|-----|--------|
| `q` / `Esc` | Quit |
| `m` | Toggle voice alerts (mute/unmute) |
| `c` | Toggle collision-risk overlay |
| `Space` | Pause/resume processing |

### Streamlit Web Demo (Local)

```bash
# Install Streamlit demo dependencies
pip install -r requirements-demo.txt

# Launch the web app
streamlit run app.py
```

Open `http://localhost:8501`, grant webcam permission, and click **START**. Use the sidebar to:
- Enable/disable voice alerts (Web Speech API)
- Adjust speech rate
- Click **Test Voice** to verify browser audio independently of the camera
- Inspect live latency metrics and obstacle telemetry

---

## Architecture

```
┌──────────┐   ┌──────────┐   ┌─────────────┐   ┌──────────────┐   ┌─────────┐
│  Input   │──▶│ Detector │──▶│  Estimator  │──▶│ Prioritizer  │──▶│ Speaker │
│ (camera/ │   │ (YOLO11  │   │ (distance + │   │ (urgency     │   │ (TTS)   │
│  image)  │   │ +ByteTrack)│  │ direction)  │   │  ranking)    │   │         │
└──────────┘   └──────────┘   └─────────────┘   └──────────────┘   └─────────┘
```

### Core Modules (`src/narrator/`)

| Module | Responsibility |
|--------|----------------|
| `detector.py` | YOLO11 inference + ByteTrack association; returns typed `Detection` objects with optional `track_id` |
| `direction.py` | Maps bbox center to discrete direction: `LEFT`, `AHEAD`, `RIGHT` |
| `distance.py` | Proximity classification (`NEAR`, `MEDIUM`, `FAR`) from bbox area heuristics |
| `depth.py` | Geometric relative depth + monocular perspective estimation |
| `ground_plane.py` | Walking-path hazard corridor via perspective trapezoid projection |
| `prioritizer.py` | Urgency scoring with path-intersection bonus |
| `events.py` | `EventNarrator`: new object, proximity escalation, urgency escalation, periodic reminders with cooldowns |
| `metrics.py` | `PerformanceMonitor`: rolling FPS, latency windows, warm-up exclusion |
| `pipeline.py` | End-to-end frame loop, HUD overlay (track IDs, zones, badges) |
| `speaker.py` | Native SAPI5/pyttsx3 async speech queue with deduplication |
| `config.py` | Typed dataclass configs (`DetectorConfig`, `EventsConfig`, etc.) + YAML/env loading |

### Web Demo Architecture (`demo/`, `app.py`)

- **Session Isolation:** Each browser connection creates an independent `NarratorVideoProcessor`, `Pipeline`, and `Detector` — ByteTrack states never cross-contaminate between visitors.
- **Client-Side Audio:** Speech synthesis runs entirely in the browser via Web Speech API. Server dispatches text via `st.html()` with `json.dumps()`-escaped payloads; no server-side audio dependencies.
- **Cloud Compatibility:** `requirements.txt` includes `lap>=0.5.12` for ByteTrack linear assignment, pre-installed during container build to avoid runtime `pip install` permission errors.

---

## Configuration

| File | Purpose |
|------|---------|
| `config/default.yaml` | Default runtime parameters (detector, speaker, events, corridor, ground, depth) |
| `pyproject.toml` | Package metadata, dependencies, pytest config |
| `requirements.txt` | Streamlit Cloud root manifest (includes `lap>=0.5.12`) |
| `requirements-demo.txt` | Mirrored demo dependencies |
| `packages.txt` | APT packages for Streamlit Cloud (`ffmpeg`) |

Environment variables override YAML defaults (see `src/narrator/config.py`):
- `NARRATOR_MODEL` — YOLO model filename (default: `yolo11s.pt`)
- `NARRATOR_DEVICE` — `cuda`, `cpu`, or `auto`

---

## Testing

```bash
# Full suite
pytest -v

# Specific areas
pytest tests/test_tracking_and_events.py -v   # ByteTrack, EventNarrator, PerformanceMonitor
pytest tests/test_web_demo.py -v              # WebRTC processor, WebSpeechBridge
pytest tests/test_ground_aware_reasoning.py -v # Walking corridor, ground plane
```

**Current Status:** 270 tests passing.

---

## Project Structure

```
Mini-Project/
├── app.py                      # Streamlit WebRTC + Web Speech demo
├── config/
│   └── default.yaml            # Default system parameters
├── demo/
│   ├── __init__.py
│   ├── processor.py            # Session-isolated WebRTC video processor
│   └── web_speaker.py          # In-memory speech bridge (no pyttsx3)
├── docs/
│   ├── architecture.md         # System design & module responsibilities
│   ├── environment.md          # CUDA/PyTorch/hardware specifications
│   └── safety_disclaimer.md    # Ethical research-prototype disclaimer
├── models/
│   └── .gitkeep                # Local YOLO weight downloads (git-ignored)
├── packages.txt                # Streamlit Cloud APT deps (ffmpeg)
├── pyproject.toml              # PEP 621 package + dev deps + pytest config
├── requirements-demo.txt       # Demo manifest
├── requirements.txt            # Cloud manifest (lap>=0.5.12 pre-installed)
├── src/
│   └── narrator/               # Core assistive vision package
│       ├── __init__.py
│       ├── __main__.py         # CLI entry point
│       ├── config.py
│       ├── depth.py
│       ├── detector.py
│       ├── diagnostics.py
│       ├── direction.py
│       ├── distance.py
│       ├── events.py
│       ├── ground_plane.py
│       ├── health.py
│       ├── metrics.py
│       ├── phrasing.py
│       ├── pipeline.py
│       ├── prioritizer.py
│       └── speaker.py
├── tests/
│   ├── __init__.py
│   ├── fixtures/
│   │   └── bus.jpg             # Standardized visual test fixture
│   ├── test_baseline_detection.py
│   ├── test_collision_risk_and_tracking.py
│   ├── test_config.py
│   ├── test_depth_and_phrasing.py
│   ├── test_ground_aware_reasoning.py
│   ├── test_ground_region.py
│   ├── test_health_and_reliability.py
│   ├── test_narrator.py
│   ├── test_relative_depth_and_spatial.py
│   ├── test_tracking_and_events.py
│   ├── test_vocabulary_and_hazards.py
│   └── test_web_demo.py
├── LICENSE                     # MIT License
└── .gitignore                  # Comprehensive exclusions
```

---

## Known Limitations

- **Missed detections:** Transparent surfaces (glass), thin objects (wires, poles), low-contrast obstacles, and ground-level hazards (curbs, steps) are frequently missed.
- **False positives:** Harmless objects may be reported as hazards; classes may be misclassified.
- **Imprecise distance:** Monocular estimation (bbox area / perspective) is uncalibrated and can be wrong by large margins.
- **Imprecise direction:** Image-relative, not true 3D bearing; depends on camera mounting.
- **Latency:** End-to-end delay between obstacle appearance and announcement.
- **Environmental sensitivity:** Low light, rain, fog, glare, and crowds degrade performance.
- **Single camera:** Limited FOV; side/rear/overhead hazards invisible.
- **Audio-only output:** No benefit if user cannot hear announcements.

---

## License

MIT License — see `LICENSE`.

**Third-party dependency notice:** Ultralytics (YOLO) is licensed under AGPL-3.0. Review their [licensing page](https://www.ultralytics.com/license) for commercial-use terms.