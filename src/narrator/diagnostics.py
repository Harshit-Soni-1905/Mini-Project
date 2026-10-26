"""Isolated detector diagnostic for Smart Obstacle Narrator.

Runs raw model.predict() inference on actual camera frames without ByteTrack,
prioritization, event logic, or speech synthesis. Exposes raw detection counts,
class names, confidences, bounding boxes, latency, and device status.
"""

from __future__ import annotations

import logging
import time

import cv2
import numpy as np

from narrator.config import AppConfig
from narrator.detector import Detector

logger = logging.getLogger(__name__)

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.55
FONT_THICKNESS = 2
HUD_FONT_SCALE = 0.6
HUD_FONT_THICKNESS = 2


def run_detector_diagnostic(config: AppConfig | None = None) -> None:
    """Run an isolated live detector diagnostic loop.

    Captures frames from the configured camera, performs raw YOLO inference,
    logs throttled detection statistics to the terminal, and renders all
    detected boxes with class names and confidences on-screen.
    """
    config = config or AppConfig()
    detector = Detector(config.detector)
    model_classes = detector.get_model_classes()

    resolved_device = config.detector.resolve_device()
    model_path = config.detector.model_path

    print("\n" + "=" * 70)
    print("=== Smart Obstacle Narrator — Isolated Detector Diagnostic ===")
    print("=" * 70)
    print(f"Model path:             {model_path}")
    print(f"Total model classes:    {len(model_classes)}")
    print(f"Device:                 {resolved_device}")
    print(f"Confidence threshold:   {config.detector.confidence_threshold:.2f}")
    print(f"Image size:             {config.detector.image_size}")
    print(f"Camera index:           {config.camera.index}")
    print("Mode:                   RAW model.predict() (no tracker, no events, no audio)")

    # Check key classes in loaded model
    key_classes = [
        "person", "bottle", "chair", "cup", "backpack", "handbag",
        "suitcase", "cell phone", "laptop", "book", "dining table",
        "bench", "bicycle", "car", "bus"
    ]
    present_classes = [c for c in key_classes if c in model_classes.values()]
    print(f"Key supported classes:  {', '.join(present_classes)}")
    print("=" * 70)
    print("Press 'q' in the diagnostic window to quit.\n")

    camera_idx = config.camera.index
    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        raise RuntimeError(
            f"Failed to open camera with index {camera_idx}. "
            "Ensure a webcam is connected and not in use by another application."
        )

    if config.camera.frame_width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.camera.frame_width)
    if config.camera.frame_height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.camera.frame_height)

    window_name = "Smart Obstacle Narrator — Isolated Detector Diagnostic"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    frame_num = 0
    last_log_time = 0.0
    log_interval = 1.0  # seconds between terminal logs

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                logger.warning("Failed to grab camera frame")
                time.sleep(0.01)
                continue

            frame_num += 1
            h, w = frame.shape[:2]

            # Run isolated raw inference
            diag = detector.diagnose_frame(frame)
            raw_dets = diag.get("detections", [])
            latency_ms = diag.get("latency_ms", 0.0)
            err = diag.get("error")

            now = time.monotonic()
            if now - last_log_time >= log_interval:
                last_log_time = now
                if err:
                    print(f"[{now:.1f}s] Frame #{frame_num}: INFERENCE ERROR: {err}")
                elif raw_dets:
                    summary = ", ".join(f"{d['label']} ({d['confidence']:.2f})" for d in raw_dets)
                    print(
                        f"[{now:.1f}s] Frame #{frame_num}: {len(raw_dets)} raw detections "
                        f"({latency_ms:.1f}ms) -> {summary}"
                    )
                else:
                    print(
                        f"[{now:.1f}s] Frame #{frame_num}: 0 detections ({latency_ms:.1f}ms) "
                        f"[conf_thresh={config.detector.confidence_threshold:.2f}]"
                    )

            # Annotate all raw detections
            canvas = frame.copy()

            for d in raw_dets:
                x1, y1, x2, y2 = int(d["x1"]), int(d["y1"]), int(d["x2"]), int(d["y2"])
                conf = d["confidence"]
                label = d["label"]

                # Cyan bounding box for raw diagnostic detections
                box_color = (255, 255, 0)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, 2)

                text = f"{label} {conf:.2f}"
                (text_w, text_h), baseline = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICKNESS)
                top_y = max(y1 - text_h - 6, 0)
                cv2.rectangle(
                    canvas,
                    (x1, top_y),
                    (x1 + text_w + 4, y1),
                    (20, 20, 20),
                    -1,
                )
                cv2.putText(
                    canvas,
                    text,
                    (x1 + 2, y1 - 4),
                    FONT,
                    FONT_SCALE,
                    (0, 255, 255),
                    FONT_THICKNESS,
                    cv2.LINE_AA,
                )

            # Top HUD banner
            overlay = canvas.copy()
            cv2.rectangle(overlay, (0, 0), (w, 42), (15, 15, 15), -1)
            cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)

            status_text = (
                f"RAW PREDICT | Detections: {len(raw_dets)} | "
                f"Inf: {latency_ms:.1f}ms | Conf: {config.detector.confidence_threshold:.2f} | "
                f"{resolved_device}"
            )
            cv2.putText(
                canvas,
                status_text,
                (10, 28),
                FONT,
                HUD_FONT_SCALE,
                (0, 255, 255),
                HUD_FONT_THICKNESS,
                cv2.LINE_AA,
            )

            cv2.imshow(window_name, canvas)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("\nQuit key received. Stopping diagnostic.")
                break
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released. Diagnostic complete.\n")
