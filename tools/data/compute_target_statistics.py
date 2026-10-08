#!/usr/bin/env python
"""Compute immutable training-only target moments on the actual loss supports."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from hydrostate.datasets.zarr_dataset import HydroStateZarrDataset, hydro_collate
from hydrostate.models.data_preprocessor import HydroDataPreprocessor
from hydrostate.models.hydrostate_model import site_supervision_values, supervision_values
from hydrostate.normalization import SUPPORT_VERSION, TASKS, training_fingerprint


class RunningMoments:
    """Merge batch moments in float64 without retaining the training labels."""

    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, values):
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        if not values.size:
            return
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite value among valid training labels")
        n = values.size
        mean = float(values.mean())
        delta = mean - self.mean
        total = self.count + n
        self.m2 += float(np.square(values - mean).sum()) + delta**2 * self.count * n / total
        self.mean += delta * n / total
        self.count = total

    def result(self):
        if self.count < 2:
            raise ValueError("At least two valid labels are required for standardization")
        std = math.sqrt(max(self.m2 / self.count, 0.0))
        if std <= 1e-8:
            raise ValueError("Near-constant target; standardization needs an explicit decision")
        return dict(count=int(self.count), mean=self.mean, std=std)


def compute_statistics(manifest, data_root, num_workers=4, batch_size=4):
    manifest = Path(manifest)
    content = manifest.read_bytes()
    table = pd.read_parquet(manifest)
    dataset = HydroStateZarrDataset(str(manifest), "train", str(data_root), training=False)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=hydro_collate,
    )
    preprocessor = HydroDataPreprocessor()
    moments = {task: RunningMoments() for task in TASKS}
    counts = {task: dict(product_count=0, site_count=0, product_samples=0) for task in TASKS}
    seen = 0
    for batch in loader:
        batch = preprocessor(batch, training=False)
        samples = batch["data_samples"]
        dummy = torch.zeros_like(samples["water"])
        for task in TASKS:
            _, values, valid = supervision_values(task, dummy, samples)
            moments[task].update(values[valid].numpy())
            counts[task]["product_count"] += int(valid.sum())
            counts[task]["product_samples"] += int(valid.reshape(len(dummy), -1).any(dim=1).sum())
            site = site_supervision_values(task, dummy, samples)
            if site is not None:
                _, values, valid = site
                moments[task].update(values[valid].numpy())
                counts[task]["site_count"] += int(valid.sum())
        seen += len(dummy)
        if seen % 128 < len(dummy) or seen == len(dataset):
            print(f"Processed {seen}/{len(dataset)} training samples", flush=True)
    if manifest.read_bytes() != content:
        raise RuntimeError("Manifest changed while computing target statistics")
    return dict(
        version=1,
        split="train",
        ddof=0,
        support_version=SUPPORT_VERSION,
        created_utc=datetime.now(UTC).isoformat(),
        source_manifest=str(manifest.resolve()),
        manifest_sha256=hashlib.sha256(content).hexdigest(),
        training_fingerprint=training_fingerprint(table),
        training_samples=len(dataset),
        tasks={task: dict(**moments[task].result(), **counts[task]) for task in TASKS},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to replace frozen statistics: {args.output}")
    report = compute_statistics(args.manifest, args.data_root, args.num_workers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents concurrent jobs replacing an existing version.
    with args.output.open("x") as handle:
        handle.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
