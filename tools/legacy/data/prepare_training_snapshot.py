#!/usr/bin/env python
"""Freeze completed GEE samples and an rsync list; extend snapshots without replacing rows."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path, PurePosixPath

import pandas as pd


def select_samples(table, limit, previous=None, seed=42):
    if table.sample_id.duplicated().any():
        raise ValueError("Source has duplicate sample IDs")
    target = len(table) if limit == 0 else min(limit, len(table))
    if target < 1:
        raise ValueError("Snapshot must contain at least one sample")
    retained = table.iloc[:0]
    if previous is not None:
        if previous.sample_id.duplicated().any():
            raise ValueError("Previous snapshot has duplicate sample IDs")
        indexed = table.set_index("sample_id", drop=False)
        missing = set(previous.sample_id) - set(indexed.index)
        if missing:
            raise ValueError(f"Previous samples disappeared from source: {sorted(missing)[:3]}")
        retained = indexed.loc[previous.sample_id].reset_index(drop=True)
        if not retained[previous.columns].equals(previous.reset_index(drop=True)):
            raise ValueError("Previously published sample records changed")
        if len(retained) > target:
            raise ValueError("An extended snapshot cannot shrink")
    candidates = table[~table.sample_id.isin(retained.sample_id)].copy()
    count = target - len(retained)
    if count:
        candidates["_rank"] = candidates.sample_id.map(
            lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()
        )
        groups = list(candidates.groupby(["split", "frequency_bin"], sort=True))
        exact = [count * len(group) / len(candidates) for _, group in groups]
        quotas = [int(value) for value in exact]
        for index in sorted(range(len(groups)), key=lambda i: (-(exact[i] - quotas[i]), i))[
            : count - sum(quotas)
        ]:
            quotas[index] += 1
        additions = pd.concat([
            group.sort_values("_rank").head(quota).drop(columns="_rank")
            for (_, group), quota in zip(groups, quotas)
        ])
        retained = pd.concat([retained, additions], ignore_index=True)
    return retained.sort_values("sample_id").reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("/mnt/d/Hydrostate/samples"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=1024, help="Total samples; 0 selects all completed samples")
    parser.add_argument("--extend", type=Path, help="Previous samples.parquet; retain every previous sample")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be nonnegative")
    source = args.source.resolve()
    content = (source / "samples.parquet").read_bytes()
    table = pd.read_parquet(io.BytesIO(content))
    previous = pd.read_parquet(args.extend) if args.extend else None
    selected = select_samples(table, args.limit, previous, args.seed)
    if set(selected.split) != {"train", "validation", "test"}:
        raise ValueError("Choose enough samples to retain train, validation and test")
    paths = sorted(set(selected.input_shard) | set(selected.label_shard))
    for value in paths:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or path.parts[0] != "zarr":
            raise ValueError(f"Not a relative Zarr path: {value}")
        resolved = (source / value).resolve()
        if not resolved.is_relative_to(source) or not (resolved / "_SUCCESS.json").is_file():
            raise ValueError(f"Missing or incomplete shard: {value}")
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = args.output / "samples.parquet"
    selected.to_parquet(manifest, index=False)
    (args.output / "zarr-files.txt").write_text("".join(value + "/\n" for value in paths))
    report = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        source=str(source), source_samples=len(table), samples=len(selected),
        seed=args.seed, extended_from=str(args.extend) if args.extend else None,
        source_manifest_sha256=hashlib.sha256(content).hexdigest(),
        manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        splits=selected.split.value_counts().to_dict(),
        frequency_bins=selected.frequency_bin.value_counts().sort_index().to_dict(),
    )
    if "labels" in selected:
        labels = selected.labels.map(json.loads)
        report["selected_product_counts"] = {
            key: int(labels.map(lambda value: bool(value.get(key))).sum())
            for key in ("dw", "sm", "pml")
        }
    (args.output / "snapshot.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
