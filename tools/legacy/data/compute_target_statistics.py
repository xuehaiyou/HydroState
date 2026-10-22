#!/usr/bin/env python3
"""Historical variable-pack statistics; not the current retrieval entry point."""

import argparse
import json
from pathlib import Path

from hydrostate.legacy.statistics import compute_statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest")
    parser.add_argument("--data-root")
    parser.add_argument("--config", help="Training config selecting tasks and variable packs")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to replace frozen statistics: {args.output}")
    options = {}
    if args.config:
        from mmengine.config import Config

        from hydrostate.cli import configure_data_contract

        cfg = Config.fromfile(args.config)
        configure_data_contract(cfg)
        dataset = cfg.train_dataloader.dataset
        if args.manifest or args.data_root:
            parser.error("Use --config or --manifest/--data-root, not both")
        args.manifest = dataset.manifest
        args.data_root = dataset.get("data_root", str(Path(args.manifest).parent))
        options = {key: dataset[key] for key in (
            "tasks", "input_packs", "label_packs", "output_scale", "target_time_queries",
            "input_profile")
                   if key in dataset}
    elif not args.manifest or not args.data_root:
        parser.error("Provide --config or both --manifest and --data-root")
    report = compute_statistics(args.manifest, args.data_root, args.num_workers, **options)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents concurrent jobs replacing an existing version.
    with args.output.open("x") as handle:
        handle.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
