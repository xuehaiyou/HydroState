"""Prepare input-only regional data, then infer on an HPC GPU allocation."""

from __future__ import annotations

import argparse
import json
import logging
import math
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
    parser.add_argument("--backend", choices=["pixels", "drive", "local"], default="pixels")
    parser.add_argument('--local-config', type=Path,
                        default=Path('configs/data/hpc_local_l2.json'))
    parser.add_argument('--template-root', type=Path,
                        help='Initialize offline using an existing plan AOI/grid/training overlap')
    parser.add_argument(
        "--download-config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs/data/gee_drive.json",
    )
    parser.add_argument("--drive-token", type=Path)
    parser.add_argument("--drive-quota-project")
    parser.add_argument(
        "--min-free-gib",
        type=float,
        default=20,
        help="Stop preparation below this free space (minimum 5 GiB)",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--model-config", type=Path,
                        help="Prediction model config; does not change the frozen input plan")
    parser.add_argument("--pretrained", type=Path)
    parser.add_argument("--statistics", type=Path)
    parser.add_argument("--drop-s2", action="store_true", help="Separate sensor ablation product")
    args = parser.parse_args()
    if (args.root / "recency_update_pending.json").exists() and args.command != "status":
        parser.error("Interrupted recency revision; rerun tools/data/update_region_recency.py")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.workers < 0 or args.batch_size < 1:
        parser.error("workers must be nonnegative and batch size positive")
    if not math.isfinite(args.min_free_gib) or args.min_free_gib < 5:
        parser.error("min-free-gib must be finite and at least 5")
    if args.command == "init":
        from hydrostate.inference.planning import initialize

        config = json.loads(Path(args.config).read_text())
        if args.template_root:
            from hydrostate.data.local.planning import initialize_from_template

            if args.pilot:
                parser.error('Template geometry is reused exactly; --pilot cannot be combined')
            plan = initialize_from_template(args.root, args.config, args.template_root)
        else:
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
        from hydrostate.data.gee.workflows.region import prepare_date, prepare_region_drive

        periods = [p for p in plan["periods"] if not args.dates or p["time"] in args.dates]
        if args.backend == 'local':
            from hydrostate.data.local.region import prepare_region_local

            prepare_region_local(args.root, plan, periods,
                                 json.loads(args.local_config.read_text()),
                                 workers=max(1, args.workers), min_free_gib=args.min_free_gib)
            return
        if args.backend == "drive":
            prepare_region_drive(
                args.root,
                plan,
                periods,
                json.loads(args.download_config.read_text()),
                workers=max(1, args.workers),
                min_free_gib=args.min_free_gib,
                drive_token=args.drive_token,
                quota_project=args.drive_quota_project,
            )
            return

        for period in periods:
            prepare_date(
                args.root,
                plan,
                period,
                workers=max(1, args.workers),
                min_free_gib=args.min_free_gib,
            )
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
            model_config=args.model_config,
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
