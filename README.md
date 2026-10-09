# Smart Obstacle Narrator

An assistive computer vision prototype that detects obstacles in a camera
feed, estimates approximate distance and direction, prioritizes warnings,
and announces concise messages through text-to-speech.

> ⚠️ **Research Prototype — Not a Safety Device**
>
> This software is an **experimental prototype** for academic and personal
> exploration.  It is **not** a substitute for a white cane, guide dog, or
> professional orientation-and-mobility training.  Object detection can miss
> obstacles — especially transparent surfaces, thin objects, and ground-level
> hazards.  Monocular distance estimation is inherently imprecise.  **Never
> rely on this system as a primary navigation aid.**

## Quick Start

### Prerequisites

- Python 3.12+ (tested with 3.14.7)
- NVIDIA GPU with CUDA support (tested with RTX 3050 Laptop, 4 GB VRAM)
- Windows 10/11 (SAPI5 TTS voices)

### Installation

```bash
# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

# Install PyTorch with CUDA support (see docs/environment.md for details)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132

# Install the project and its dependencies
pip install -e ".[dev]"
```

### Verify the setup

```bash
# Run the test suite
pytest

# Run the baseline detection test on a sample image
python tests/test_baseline_detection.py
```

## Project Structure

```
smart-obstacle-narrator/
├── src/narrator/          # Main application package
│   ├── config.py          # Configuration and constants
│   ├── detector.py        # Object detection wrapper
│   ├── direction.py       # Directional mapping (future)
│   ├── distance.py        # Distance estimation (future)
│   ├── pipeline.py        # Processing pipeline (future)
│   ├── prioritizer.py     # Hazard prioritization (future)
│   └── speaker.py         # Text-to-speech wrapper (future)
├── config/                # Runtime configuration files
├── scripts/               # Entry-point scripts
├── tests/                 # Test suite
│   └── fixtures/          # Test images and data
├── notebooks/             # Exploration notebooks
├── models/                # Downloaded model weights (git-ignored)
├── data/                  # Datasets (git-ignored)
└── docs/                  # Documentation
```

## Architecture

See [docs/architecture.md](docs/architecture.md) for the system design.

## Environment

See [docs/environment.md](docs/environment.md) for tested versions and
PyTorch/CUDA installation details.

## License

This project is licensed under the MIT License.  Note that the Ultralytics
library (YOLO) is licensed under AGPL-3.0 — see their
[licensing page](https://www.ultralytics.com/license) for commercial-use
terms.

## Safety Disclaimer

See [docs/safety_disclaimer.md](docs/safety_disclaimer.md) for the full
research-prototype disclaimer.
