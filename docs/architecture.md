# Architecture Overview

## System Design

Smart Obstacle Narrator follows a pipeline architecture where each stage
is an independent, testable module.

```
┌─────────┐    ┌──────────┐    ┌──────────┐    ┌────────────┐    ┌─────────┐
│  Input   │───▶│ Detector │───▶│ Estimator│───▶│ Prioritizer│───▶│ Speaker │
│ (camera/ │    │ (YOLO)   │    │(distance │    │ (rank by   │    │ (TTS)   │
│  image)  │    │          │    │+direction│    │  urgency)  │    │         │
└─────────┘    └──────────┘    └──────────┘    └────────────┘    └─────────┘
```

## Modules

### `narrator.detector`
Wraps the Ultralytics YOLO model.  Loads a pretrained model (default:
`yolo11n.pt`), runs inference on a frame, and returns structured detection
results (class, confidence, bounding box coordinates).

### `narrator.direction`
Maps each detection's bounding-box center to an image-relative direction
label (e.g., "left", "ahead", "right").  Uses the horizontal position
within the frame — not true 3D bearing.

### `narrator.distance`
Estimates approximate distance from the camera.  The initial approach uses
bounding-box area as a rough proxy (larger box → closer object).  A future
stage may integrate monocular depth estimation if the heuristic proves
insufficient.

### `narrator.prioritizer`
Ranks detections by urgency, considering estimated proximity and object
class.  Objects that are both close and potentially dangerous (vehicles,
cyclists) are announced before distant or benign ones.

### `narrator.speaker`
Wraps the text-to-speech engine (pyttsx3 / SAPI5 on Windows).  Manages
speech rate, queuing, cooldowns, and deduplication so the user is not
overwhelmed with repeated announcements.

### `narrator.pipeline`
Orchestrates the full loop: capture a frame, detect, estimate, prioritize,
speak.  Handles frame-rate throttling and graceful shutdown.

### `narrator.config`
Central configuration: model paths, confidence thresholds, device
selection, class filters, speech parameters.  Loads from
`config/default.yaml` with environment-variable overrides.

## Design Principles

1. **Modularity** — each module has a single responsibility and can be
   unit-tested in isolation.
2. **Pretrained first** — use off-the-shelf models before considering
   fine-tuning.
3. **Graceful degradation** — fall back to CPU if no GPU is available;
   fall back to simpler estimation if a model fails to load.
4. **Honest limitations** — never claim precision that the system cannot
   deliver.  Distance and direction estimates are approximate.
5. **4 GB VRAM budget** — all inference must fit comfortably in the
   available GPU memory.
