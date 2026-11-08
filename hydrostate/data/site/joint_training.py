"""Read-only joint-training preflight; no publication or optimizer steps."""

import json
from pathlib import Path

from mmengine.config import Config
from mmengine.registry import init_default_scope

from hydrostate.cli import configure_target_statistics
from hydrostate.data.gee.contracts import atomic_json
from hydrostate.datasets.collate import hydro_collate
from hydrostate.registry import DATASETS

from .archives import sha256


def verify_joint_training(config, output, *, check_batch=False):
    cfg = Config.fromfile(str(config))
    init_default_scope("hydrostate")
    configure_target_statistics(cfg, training=True)
    result = dict(
        status="passed", scope="configuration_and_contracts", config=str(config), splits={}
    )
    for split, name in [("train", "train"), ("validation", "val"), ("test", "test")]:
        dataset = DATASETS.build(dict(cfg[f"{name}_dataloader"].dataset))
        result["splits"][split] = dict(
            products=len(dataset.product), sites=sum(map(len, dataset.sites))
        )
        if split == "train":
            weights = dataset.sampling_weights()
            sites = []
            for i, component in enumerate(dataset.sites, start=1):
                for row_id, row in component.table.iterrows():
                    sites.append(
                        (int(row.frequency_bin), float(weights[dataset.offsets[i] + row_id]))
                    )
            result["sampling_mass"] = dict(
                product=float(weights[: len(dataset.product)].sum()),
                site_by_frequency={
                    str(b): sum(w for k, w in sites if k == b)
                    for b in sorted({b for b, _ in sites})
                },
                interpretation="Expected sampling mass; repeated draws are not independent data",
            )
        if check_batch:
            indices = [0]
            for i, site in enumerate(dataset.sites, start=1):
                for mask in (
                    site.table.valid_sm.gt(0),
                    site.table.et_support.eq("static_210m"),
                    site.table.et_support.eq("footprint"),
                ):
                    if mask.any():
                        indices.append(dataset.offsets[i] + int(mask[mask].index[0]))
            indices = list(dict.fromkeys(indices))
            batch = hydro_collate([dataset[i] for i in indices])
            result["splits"][split]["checked_batch"] = len(indices)
            result["splits"][split]["site_labels_in_batch"] = {
                task: int(batch["data_samples"][f"site_{task}_native_valid"].sum())
                for task in ("sm", "et")
            }
    root = Path(cfg.site_split_root)
    result["site_artifacts"] = {
        name: sha256(root / name)
        for name in ("split_report.json", "site_target_stats.json", "inventory.parquet")
    }
    result["statistics"] = dict(
        product=cfg.model.target_statistics["training_fingerprint"],
        site=cfg.model.site_supervision.statistics,
    )
    if check_batch:
        result["scope"] += "_and_actual_mixed_Zarr_batches_no_model_step"
    atomic_json(Path(output), result)
    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--check-batch", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(verify_joint_training(args.config, args.output, check_batch=args.check_batch)),
        flush=True,
    )
