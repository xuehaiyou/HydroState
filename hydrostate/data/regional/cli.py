"""Formal regional preparation and inference commands."""

import argparse
import json
import logging
from pathlib import Path


def prepare_main():
    from hydrostate.data.regional.pipeline import prepare, status
    from hydrostate.data.regional.plan import initialize

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "prepare", "status", "audit"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--drive-token", type=Path)
    parser.add_argument("--retry-failed", action="store_true")
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
        result = prepare(args.root, token=args.drive_token, retry_failed=args.retry_failed)
    else:
        result = status(args.root, audit=args.command == "audit")
    print(json.dumps(result, indent=2))


def predict_main():
    from hydrostate.inference.regional import load_model, predict

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
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    model, provenance = load_model(
        args.model_config, args.checkpoint, args.pretrained, args.statistics, args.device
    )
    result = predict(
        args.root,
        model,
        provenance,
        dates=args.dates,
        batch_size=args.batch_size,
        workers=args.workers,
        device=args.device,
    )
    print(json.dumps(dict(product=result), indent=2))
