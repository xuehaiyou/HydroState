#!/usr/bin/env python
"""Validate a real ecosystem pack against legacy water and training support."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from hydrostate.data.gee_labels import COVER_NAMES, NATIVE_NAMES, tasks
from hydrostate.datasets.collate import hydro_collate
from hydrostate.legacy.dataset import HydroStateZarrDataset
from hydrostate.models.data_preprocessor import HydroDataPreprocessor
from hydrostate.models.hydrostate_model import combined_supervision_values, prepare_supervision


def validate_training_support(outputs, samples, specs):
    """Compare the two training paths on both product and optional station support."""
    prepared = prepare_supervision(outputs, samples, tasks=specs)
    loss = sum(pred.sum() * 0 for pred in outputs.values())
    counts = {}
    for name, plan in prepared.items():
        # Station fields are present even when every station observation is masked.
        # Training still appends those slots, so product-only tensors cannot be
        # compared to the flattened product-plus-station projection.
        pred, target, valid = combined_supervision_values(
            name, outputs[name], samples, tasks=specs)
        torch.testing.assert_close(pred, plan.project(outputs[name]))
        torch.testing.assert_close(target, plan.target)
        torch.testing.assert_close(valid, plan.valid)
        counts[name] = int(valid.sum())
        assert counts[name] == int(plan.count)
        if valid.any():
            assert torch.isfinite(target[valid]).all()
            loss = loss + (pred[valid] - target[valid]).square().mean()
    return loss, counts


def validate(manifest, pack, limit=64):
    table = pd.read_parquet(manifest)
    index = pd.read_parquet(pack / "index.parquet")
    selected = set(index.sample_id.iloc[:limit] if limit else index.sample_id)
    specs = tasks()
    counts = dict.fromkeys(specs, 0)
    checked = 0
    for split in table.split.unique():
        dataset = HydroStateZarrDataset(
            str(manifest), split, tasks=specs,
            label_packs=[dict(path=str(pack.resolve()), max_uncertainty=dict(lst=2.0))])
        positions = [i for i, sid in enumerate(dataset.table.sample_id) if sid in selected]
        for offset in range(0, len(positions), 4):
            items = []
            for i in positions[offset:offset + 4]:
                item = dataset[i]
                targets = item["data_samples"]
                mask = targets["water_valid"]
                for name in COVER_NAMES:
                    torch.testing.assert_close(targets[name + "_valid"], mask)
                total = sum(targets[name] for name in COVER_NAMES)
                torch.testing.assert_close(total[mask], torch.ones_like(total[mask]))
                row = dataset.table.iloc[i]
                legacy = Path(row.label_shard)
                if not legacy.is_absolute():
                    legacy = manifest.parent / legacy
                group = dataset._cache.get(str(legacy))
                np.testing.assert_array_equal(targets["water"].numpy(),
                                              group["water"][int(row.label_row)])
                np.testing.assert_array_equal(mask.numpy(),
                                              group["water_valid"][int(row.label_row)])
                assert not (targets["lst_valid"] & (targets["lst_uncertainty"] > 2)).any()
                for name in NATIVE_NAMES:
                    weights = targets[f"{name}_native_weights"]
                    valid = targets[f"{name}_native_valid"]
                    assert torch.isfinite(weights).all() and (weights >= 0).all()
                    torch.testing.assert_close(weights[valid].sum((-1, -2)),
                                               torch.ones(int(valid.sum())), atol=1e-5, rtol=0)
                items.append(item)
            batch = HydroDataPreprocessor(tasks=specs)(hydro_collate(items))
            samples = batch["data_samples"]
            outputs = {name: torch.ones_like(samples["prediction_valid"], dtype=torch.float32,
                                             requires_grad=True) for name in specs}
            loss, batch_counts = validate_training_support(outputs, samples, specs)
            for name, count in batch_counts.items():
                counts[name] += count
            loss.backward()
            assert torch.isfinite(loss)
            assert all(torch.isfinite(pred.grad).all() for pred in outputs.values())
            checked += len(items)
            print(f"Validated {checked}/{len(selected)} samples ({split})", flush=True)
    if checked != len(selected) or not checked:
        raise ValueError("Pack sample IDs did not resolve to the base manifest")
    return dict(checked_samples=checked, pack=str(pack.resolve()), lst_threshold_k=2,
                water_matches_legacy=True, cover_closure=True, native_weights=True,
                training_support_and_backward=True, valid_training_support_counts=counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=64, help="0 checks every indexed sample")
    parser.add_argument("--report", type=Path,
                        default=Path("outputs/gee_labels_validation/report.json"))
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("limit cannot be negative")
    report = validate(args.manifest, args.pack, args.limit)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
