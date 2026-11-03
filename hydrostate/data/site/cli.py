"""Explicit, resumable station data preparation stages."""

import argparse
import json
from pathlib import Path

from hydrostate.data.gee.contracts import atomic_json

from .archives import extract, inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    command = stages.add_parser("index", help="Index source headers without scanning observations")
    command.add_argument("--root", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--limit", type=int)
    for name in ("inventory", "extract"):
        command = stages.add_parser(name)
        command.add_argument("--root", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--pattern", default="*/*.zip")
    command = stages.add_parser("labels", help="Prepare subdaily window labels, before spatial QC")
    command.add_argument("--root", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--pattern", required=True)
    command.add_argument("--source", choices=["ismn", "fluxnet"], required=True)
    command.add_argument("--start", required=True)
    command.add_argument("--end", required=True, help="Exclusive UTC end")
    command.add_argument("--as-of", required=True, help="Frozen ended-context ceiling (UTC)")
    command.add_argument("--limit", type=int, help="Bounded pilot file count")
    command = stages.add_parser("pilot", help="Collect unreleased SM pilot Zarrs")
    command.add_argument("--label-record", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--config", type=Path, required=True)
    command.add_argument("--target-times", nargs="+", required=True)
    command = stages.add_parser("compare-sm", help="SMAP pairing diagnostic for development Zarrs")
    command.add_argument("--manifest", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--project", required=True)
    command = stages.add_parser("validate-model", help="CPU forward/backward on real station Zarrs")
    command.add_argument("--manifest", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--config", type=Path, required=True)
    command.add_argument("--product-statistics", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "index":
        from .index import build_index

        print(json.dumps(build_index(args.root, args.output, limit=args.limit)), flush=True)
        return
    if args.stage == "validate-model":
        from .validation import validate_training_path

        validate_training_path(args.manifest, args.config, args.product_statistics, args.output)
        return
    if args.stage == "compare-sm":
        from hydrostate.data.gee.client import initialize

        from .comparison import collect_sm_comparison, plot_sm_comparison

        report = collect_sm_comparison(initialize(args.project), args.manifest, args.output)
        plot_sm_comparison(report, args.output)
        print(json.dumps(report["overall"]), flush=True)
        return
    if args.stage == "pilot":
        from mmengine.config import Config
        from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.data.normalize import (
            load_computed_config,
        )

        from hydrostate.data.gee.client import initialize

        from .pilot import collect_pilot

        config = Config.fromfile(str(args.config)).to_dict()
        record = json.loads(args.label_record.read_text())
        selected = [r for r in record["windows"] if r["target_time"] in args.target_times]
        if len(selected) != len(set(args.target_times)):
            parser.error("Some requested target windows are absent from the label record")
        from .temporal import DAY, epoch

        starts = sorted(epoch(r["target_time"]) for r in selected)
        if any(b-a < config["min_date_gap_days"]*DAY for a, b in zip(starts, starts[1:])):
            parser.error("Pilot target windows violate the configured per-station date gap")
        collect_pilot(initialize(config["project"]), selected, args.output, config,
                       load_computed_config())
        return
    paths = sorted(args.root.glob(args.pattern))
    if not paths:
        parser.error("No source archives matched")
    if args.stage == "labels":
        from .prepare import prepare_labels

        if args.limit is not None:
            if args.limit <= 0:
                parser.error("--limit must be positive")
            paths = paths[:args.limit]
        prepare_labels(paths, args.output, args.source, args.start, args.end, args.as_of)
        return
    if args.stage == "inventory":
        records = inventory(paths)
    else:
        records = []
        for index, path in enumerate(paths, 1):
            record = extract(path)
            records.append(record)
            atomic_json(args.output, dict(complete=False, archives=records))
            print(json.dumps(dict(index=index, total=len(paths), **record)), flush=True)
    atomic_json(args.output, dict(complete=True, archives=records))


if __name__ == "__main__":
    main()
