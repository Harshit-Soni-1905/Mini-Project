"""Baseline detection test — standalone smoke test for YOLO inference.

Relocated and improved from the original test_detection.py.
Can be run directly: python tests/test_baseline_detection.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Ensure src/ is importable when run as a script.
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "src"))


def main() -> None:
    import torch
    from ultralytics import YOLO

    print("=== Smart Obstacle Narrator — Baseline Detection Test ===\n")

    # --- Device selection ---
    if torch.cuda.is_available():
        device = "0"
        print(f"CUDA device : {torch.cuda.get_device_name(0)}")
        print(f"VRAM         : {torch.cuda.get_device_properties(0).total_memory // (1024 ** 2)} MB")
    else:
        device = "cpu"
        print("CUDA         : not available, using CPU")

    # --- Model ---
    model = YOLO("yolo11n.pt")
    print(f"Model        : yolo11n.pt ({sum(p.numel() for p in model.model.parameters()) / 1e6:.1f}M params)")

    # --- Source image ---
    fixture = Path(__file__).resolve().parent / "fixtures" / "bus.jpg"
    if fixture.exists():
        source = str(fixture)
        print(f"Source       : {source}")
    else:
        source = "https://ultralytics.com/images/bus.jpg"
        print(f"Source       : {source}  (fixture not found, using URL)")

    # --- Inference ---
    print("\nRunning inference...")
    t0 = time.perf_counter()

    results = model.predict(
        source=source,
        device=device,
        imgsz=640,
        conf=0.35,
        save=False,  # don't clutter the project with runs/
        verbose=False,
    )

    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"Inference    : {elapsed_ms:.0f} ms (includes model warmup)\n")

    # --- Results ---
    print(f"{'Class':<20} {'Confidence':>10}   {'Box (x1 y1 x2 y2)'}")
    print("-" * 65)

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            label = model.names[class_id]
            coords = box.xyxy[0].tolist()
            print(f"{label:<20} {confidence:>10.2%}   "
                  f"[{coords[0]:.0f}, {coords[1]:.0f}, {coords[2]:.0f}, {coords[3]:.0f}]")

    total = sum(len(r.boxes) for r in results)
    print(f"\nTotal detections: {total}")
    print("\n✓ Baseline detection test complete.")


if __name__ == "__main__":
    main()
