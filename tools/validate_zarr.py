#!/usr/bin/env python
"""Audit the current retrieval manifest, all shard contracts, and sampled pixels."""

import argparse
import json
from pathlib import Path

from mmengine.config import Config

from hydrostate.cli import configure_data_contract
from hydrostate.data.retrieval.audit import audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="Training config or collection directory")
    parser.add_argument(
        "--pixel-limit", type=int, default=64, help="Array checks: 0 checks all samples"
    )
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    if args.pixel_limit < 0:
        parser.error("--pixel-limit must be nonnegative")
    options = {}
    if args.source.endswith(".py"):
        cfg = Config.fromfile(args.source)
        configure_data_contract(cfg)
        root = Path(cfg.train_dataloader.dataset.data_root)
        options = dict(label_packs=cfg.get("label_packs"), label_specs=cfg.get("label_specs"))
    else:
        root = Path(args.source)
    print(
        json.dumps(
            audit(
                root,
                pixel_limit=args.pixel_limit,
                require_complete=not args.allow_partial,
                **options,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
