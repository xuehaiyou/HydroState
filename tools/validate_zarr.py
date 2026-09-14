#!/usr/bin/env python
"""Validate manifest references and sampled Zarr schemas before training."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import zarr
from mmengine.config import Config

from hydrostate.storage import Schema


def _paths_from_config(path: str) -> tuple[Path, Path]:
    cfg = Config.fromfile(path)
    dataset = cfg.train_dataloader.dataset
    return Path(dataset.manifest), Path(dataset.data_root)


def _resolve(value: str, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _validate_group(path: Path, specs: dict, expected_length: int) -> list[str]:
    errors = []
    if not path.exists():
        return [f"missing shard: {path}"]
    if not (path / "_SUCCESS.json").exists():
        errors.append(f"incomplete shard (no _SUCCESS.json): {path}")
    group = zarr.open_group(str(path), mode="r")
    for name, spec in specs.items():
        if name not in group:
            errors.append(f"{path}: missing array {name}")
            continue
        array = group[name]
        if tuple(array.shape[1:]) != spec.tail_shape:
            errors.append(
                f"{path}/{name}: tail shape {array.shape[1:]} != {spec.tail_shape}"
            )
        if array.shape[0] < expected_length:
            errors.append(f"{path}/{name}: only {array.shape[0]} rows, need {expected_length}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="MMEngine config.py or samples.parquet")
    parser.add_argument("--data-root")
    args = parser.parse_args()
    if args.source.endswith(".py"):
        manifest_path, data_root = _paths_from_config(args.source)
    else:
        manifest_path = Path(args.source)
        data_root = Path(args.data_root or manifest_path.parent)

    table = pd.read_parquet(manifest_path)
    required = {"sample_id", "split", "input_shard", "input_row", "label_shard", "label_row"}
    missing = required.difference(table.columns)
    if missing:
        raise SystemExit(f"manifest missing columns: {sorted(missing)}")
    if table.sample_id.duplicated().any():
        raise SystemExit("manifest contains duplicate sample_id values")

    schema = Schema()
    errors: list[str] = []
    for column, row_column, specs in (
        ("input_shard", "input_row", schema.inputs),
        ("label_shard", "label_row", schema.targets),
    ):
        for shard, rows in table.groupby(column)[row_column]:
            expected_length = int(rows.max()) + 1
            errors.extend(_validate_group(_resolve(str(shard), data_root), specs, expected_length))
    if errors:
        print("\n".join(errors[:100]))
        raise SystemExit(f"validation failed with {len(errors)} error(s)")
    print(f"validated {len(table)} samples across {table.input_shard.nunique()} input shards")


if __name__ == "__main__":
    main()

