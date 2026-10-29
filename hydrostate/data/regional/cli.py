"""Formal regional preparation and inference commands."""

import argparse
import json
import logging
from pathlib import Path


def prepare_main():
    from hydrostate.data.regional.pipeline import prepare as prepare_drive
    from hydrostate.data.regional.pipeline import status
    from hydrostate.data.regional.pixels import prepare as prepare_pixels
    from hydrostate.data.regional.plan import initialize

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "prepare", "status", "audit"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--drive-token", type=Path)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--backend", choices=("pixels", "drive"), default="pixels")
    parser.add_argument("--drain", action="store_true", help="Finish existing Drive exports only")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.command == "plan":
        if args.config is None:
            parser.error("plan requires --config")
        result = initialize(args.root, json.loads(args.config.read_text()))
        result = dict(
            root=str(args.root),
            identity=result["identity"],
            grid=result["grid"],
            target_periods=len(result["periods"]),
        )
    elif args.command == "prepare":
        if args.config:
            initialize(args.root, json.loads(args.config.read_text()))
        if args.backend == "pixels":
            if args.drain or args.retry_failed or args.drive_token:
                parser.error("--drain/--retry-failed/--drive-token require --backend drive")
            result = prepare_pixels(args.root)
        else:
            result = prepare_drive(
                args.root,
                token=args.drive_token,
                retry_failed=args.retry_failed,
                drain_only=args.drain,
            )
    else:
        result = status(args.root, audit=args.command == "audit")
        progress = args.root / "pixels_progress.json"
        if progress.exists():
            result["pixels_progress"] = json.loads(progress.read_text())
    print(json.dumps(result, indent=2))


def predict_main():
    from hydrostate.inference.regional_runtime import run

    parser = argparse.ArgumentParser(description="Predict current seven-task regional inputs")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--statistics", required=True)
    parser.add_argument("--dates", nargs="+")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--batch-candidates", type=int, nargs="+", help="Measure CUDA batches first"
    )
    parser.add_argument("--pilot-dates", nargs="+", help="Predict and preview these dates first")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--progress-seconds", type=float, default=60)
    parser.add_argument("--display-surface-min", type=float, default=0.05)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(json.dumps(run(args), indent=2))
