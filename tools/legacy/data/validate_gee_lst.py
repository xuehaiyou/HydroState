#!/usr/bin/env python
"""Check actual LST times, paired inputs, fixed QA and downstream training support."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from hydrostate.data.gee_labels import tasks
from hydrostate.data.gee_lst import input_scene_pairs
from hydrostate.data.target_times import epoch_seconds
from hydrostate.datasets.collate import hydro_collate
from hydrostate.storage.reader import ShardCache
from hydrostate.legacy.dataset import HydroStateZarrDataset
from hydrostate.models.data_preprocessor import HydroDataPreprocessor
from hydrostate.storage.packs import VariablePack

if __package__:
    from .validate_gee_labels import validate_training_support
else:
    from validate_gee_labels import validate_training_support


class SupportDataset(HydroStateZarrDataset):
    """Support validation needs real masks/times, not the satellite pixel values."""

    def _load_inputs(self, row):
        path = Path(row.input_shard)
        group = self._cache.get(str(path if path.is_absolute() else self.data_root / path))
        index = int(row.input_row)
        inputs = {name: torch.from_numpy(np.asarray(group[name][index]))
                  for name in ("sensor_valid", "observation_time")}
        for name, channels in (("s1", 2), ("s2", 12), ("landsat", 11)):
            mask = torch.from_numpy(np.asarray(group[name + "_valid"][index], bool))
            inputs[name + "_valid"] = mask
            inputs[name] = torch.zeros(mask.shape[0], channels, *mask.shape[-2:])
        return inputs


def validate(manifest, baseline, pack, limit=64, training_limit=64):
    table = pd.read_parquet(manifest)
    rows = table.set_index("sample_id", drop=False)
    index = pd.read_parquet(pack / "index.parquet")
    selected = index.iloc[:limit] if limit else index
    old = VariablePack(baseline, table, kind="label", variables=["lst"])
    new = VariablePack(pack, table, kind="label", variables=["lst"])
    cache, methods, splits = ShardCache(), Counter(), Counter()
    old_count, new_count, pixels, offsets = 0, 0, 0, []
    for observation_index, entry in enumerate(selected.itertuples(), 1):
        row = rows.loc[entry.sample_id]
        before = old.read(entry.sample_id, cache)
        after = new.read(entry.sample_id, cache)
        provenance = json.loads(entry.provenance)
        mask = after["lst_valid"]
        old_count += int(before["lst_valid"].any())
        new_count += int(mask.any())
        pixels += int(mask.sum())
        methods[provenance["selection"]] += 1
        if mask.any():
            scene = provenance["scenes"]["lst"]
            stamp = int(scene["time_ms"]) // 1000
            np.testing.assert_array_equal(after["lst_time"], [stamp, stamp])
            assert np.isfinite(after["lst"][mask]).all()
            assert (after["lst_uncertainty"][mask] <= 2).all()
            assert (after["lst_uncertainty"][mask] >= 0).all()
            splits[row.split] += 1
            offsets.append((stamp - epoch_seconds(row.target_time)) / 86400)
            if provenance["selection"] == "paired_input":
                pairs = input_scene_pairs(dict(inputs=json.loads(row.inputs)))
                paired = pairs[scene["asset_id"]]
                assert paired["input_time_ms"] == scene["time_ms"]
                assert paired["input_slot"] == scene["input_slot"]
                shard = Path(row.input_shard)
                group = cache.get(str(shard if shard.is_absolute() else manifest.parent / shard))
                slot = paired["input_slot"]
                assert group["sensor_valid"][row.input_row, slot, 2]
                assert int(group["observation_time"][row.input_row, slot, 2]) == stamp
                support = group["landsat_valid"][row.input_row, slot]
                support = support.reshape(1, 40, 3, 40, 3).all(axis=(2, 4))
                assert (support | ~mask).all()
            elif provenance["selection"] == "preserved_same_day":
                assert (before["lst_valid"] | ~mask).all()
                np.testing.assert_array_equal(after["lst"][mask], before["lst"][mask])
        else:
            assert not after["lst_time"].any()
        if observation_index % 64 == 0:
            print(f"Verified observation {observation_index}/{len(selected)}", flush=True)

    specs = tasks()
    variables = [name for name in old.metadata["variables"] if name != "lst"]
    packs = [dict(path=str(baseline.resolve()), variables=variables),
             dict(path=str(pack.resolve()), variables=["lst"], max_uncertainty=dict(lst=2.0))]
    print(f"Verified observations and QA for {len(selected)} samples", flush=True)
    training = selected.iloc[:training_limit] if training_limit else selected
    ids = set(training.sample_id)
    counts = Counter()
    checked = 0
    for split in table.split.unique():
        dataset = SupportDataset(str(manifest), split, tasks=specs,
                                 label_packs=packs, target_time_queries=True)
        positions = [i for i, sid in enumerate(dataset.table.sample_id) if sid in ids]
        for start in range(0, len(positions), 4):
            items = [dataset[i] for i in positions[start:start + 4]]
            batch = HydroDataPreprocessor(tasks=specs)(hydro_collate(items))
            samples, inputs = batch["data_samples"], batch["inputs"]
            for name, bounds in inputs["target_time"].items():
                assert bounds.dtype == torch.long
                assert ((bounds[:, 0] > 0) & (bounds[:, 1] >= bounds[:, 0])).all()
            valid = samples["lst_valid"].flatten(1).any(1)
            torch.testing.assert_close(inputs["target_time"]["lst"][valid],
                                       samples["lst_time"][valid])
            outputs = {name: torch.ones_like(samples["prediction_valid"], dtype=torch.float32,
                                             requires_grad=True) for name in specs}
            loss, batch_counts = validate_training_support(outputs, samples, specs)
            counts.update(batch_counts)
            loss.backward()
            assert torch.isfinite(loss)
            assert all(torch.isfinite(pred.grad).all() for pred in outputs.values())
            checked += len(items)
            if checked % 32 < len(items):
                print(f"Validated {checked}/{len(ids)} samples ({split})", flush=True)
        print(f"Validated {checked}/{len(ids)} samples ({split})", flush=True)
    assert checked == len(training) and checked
    return dict(checked_samples=len(selected), training_checked_samples=checked,
                baseline_lst_samples=old_count,
                new_lst_samples=new_count, new_lst_pixels_30m=pixels,
                selection_counts=dict(methods), lst_samples_by_split=dict(splits),
                observation_offset_days=(dict(min=min(offsets), max=max(offsets))
                                         if offsets else None),
                lst_threshold_k=2, paired_input_identity_and_time=True,
                query_times_and_training_support=True, valid_training_support_counts=dict(counts))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base-pack", type=Path, required=True)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=64, help="0 checks every indexed sample")
    parser.add_argument("--training-limit", type=int, default=64,
                        help="Subset for all-task supervision/backward checks; 0 checks all")
    parser.add_argument("--report", type=Path,
                        default=Path("outputs/gee_lst_validation/report.json"))
    args = parser.parse_args()
    if min(args.limit, args.training_limit) < 0:
        parser.error("limits cannot be negative")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    report = validate(args.manifest, args.base_pack, args.pack, args.limit, args.training_limit)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
