"""Prepare input-only regional data, then infer on an HPC GPU allocation."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["init", "prepare", "predict", "status"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", default="configs/inference/poyang_3vars.json")
    parser.add_argument("--training-manifest", type=Path)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--dates", nargs="+", help="Exact nominal dates listed in plan.json")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--pretrained", type=Path)
    parser.add_argument("--statistics", type=Path)
    parser.add_argument("--drop-s2", action="store_true", help="Separate sensor ablation product")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.workers < 0 or args.batch_size < 1:
        parser.error("workers must be nonnegative and batch size positive")
    if args.command == "init":
        from hydrostate.inference.planning import initialize

        config = json.loads(Path(args.config).read_text())
        plan = initialize(
            args.root,
            args.config,
            args.training_manifest or config["training_manifest"],
            pilot=args.pilot,
        )
    else:
        plan = json.loads((args.root / "plan.json").read_text())
    available = {p["time"] for p in plan["periods"]}
    if args.dates and not set(args.dates).issubset(available):
        parser.error(f"Dates must be in plan.json; available: {sorted(available)}")
    if args.command == "prepare":
        from hydrostate.inference.gee import prepare_date

        for period in plan["periods"]:
            if args.dates and period["time"] not in args.dates:
                continue
            prepare_date(args.root, plan, period, workers=max(1, args.workers))
    elif args.command == "predict":
        from hydrostate.inference.predict import predict_dates

        predict_dates(
            args.root,
            plan,
            args.dates,
            batch_size=args.batch_size,
            workers=args.workers,
            device=args.device,
            checkpoint=args.checkpoint,
            pretrained=args.pretrained,
            statistics=args.statistics,
            drop_s2=args.drop_s2,
        )
    else:
        from hydrostate.inference.planning import windows

        print(
            json.dumps(
                dict(
                    root=str(args.root),
                    area_km2=plan["area_km2"],
                    grid=plan["grid"],
                    dates=[p["time"] for p in plan["periods"]],
                    windows_per_date=len(windows(plan)),
                    overlapping_training_records=len(plan["training"]),
                    prepared_dates=sorted(
                        p.stem for p in (args.root / "manifests").glob("*.parquet")
                    ),
                ),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
