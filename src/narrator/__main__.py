"""Command-line entry point for Smart Obstacle Narrator.

Run with:
    python -m narrator
"""

from __future__ import annotations

import argparse
import logging
import sys

from narrator.config import AppConfig
from narrator.pipeline import Pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smart Obstacle Narrator — real-time assistive obstacle detection.",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=None,
        help="OpenCV camera index (default: 0, or NARRATOR_CAMERA env var)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help="Confidence threshold (default: 0.35, or NARRATOR_CONFIDENCE env var)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Inference device ('0' for CUDA, 'cpu' for CPU; default: auto-select)",
    )
    parser.add_argument(
        "--mute",
        action="store_true",
        help="Disable text-to-speech narration",
    )
    parser.add_argument(
        "--no-track",
        action="store_true",
        help="Disable persistent object tracking (use raw frame-by-frame prediction)",
    )
    parser.add_argument(
        "--tracker",
        type=str,
        default=None,
        help="Tracker configuration file (e.g., 'bytetrack.yaml', 'botsort.yaml')",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug-level logging",
    )
    parser.add_argument(
        "--debug-detections",
        action="store_true",
        help="Enable detailed detection diagnostics (print raw detections, filtered results, etc.)",
    )
    parser.add_argument(
        "--diagnose-detector",
        action="store_true",
        help="Run isolated detector diagnostic (raw model inference without tracking, events, or audio)",
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Build config from defaults + env vars + CLI flags
    config = AppConfig.from_env()
    if args.camera is not None:
        config.camera.index = args.camera
    if args.conf is not None:
        config.detector.confidence_threshold = args.conf
    if args.device is not None:
        config.detector.device = args.device
    if args.mute:
        config.speaker.enabled = False
    if args.no_track:
        config.detector.tracking_enabled = False
    if args.tracker is not None:
        config.detector.tracker_type = args.tracker
    if args.debug_detections:
        config.debug_detections = True

    if args.diagnose_detector:
        from narrator.diagnostics import run_detector_diagnostic
        try:
            run_detector_diagnostic(config)
        except KeyboardInterrupt:
            print("\nDiagnostic shutdown requested by user.")
        except Exception as exc:
            print(f"\nDiagnostic Error: {exc}", file=sys.stderr)
            sys.exit(1)
        return

    try:
        pipeline = Pipeline(config)
        pipeline.run()
    except KeyboardInterrupt:
        print("\nShutdown requested by user.")
    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
