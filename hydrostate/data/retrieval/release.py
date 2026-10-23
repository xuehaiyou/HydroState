"""Prepare and verify a completed dataset for the paired MTLoRA/LoRA runs."""

import argparse
import json
import os
from pathlib import Path

import pandas as pd
from mmengine.config import Config

from hydrostate.cli import configure_data_contract, configure_target_statistics
from hydrostate.data.gee.contracts import atomic_json
from hydrostate.datasets.retrieval_dataset import RetrievalDataset
from hydrostate.normalization import training_fingerprint
from hydrostate.storage.packs import file_hash

from .audit import audit
from .contract import PROFILE, SENSORS, TIME_STEPS, VERSION
from .statistics import compute_retrieval_statistics


def verify(root, statistics):
    root, statistics = Path(root), Path(statistics)
    marker = json.loads((root / "_COLLECTION_SUCCESS.json").read_text())
    if (
        not marker.get("complete")
        or not marker.get("completed_at")
        or marker.get("version") != VERSION
        or marker.get("input_profile") != PROFILE
    ):
        raise ValueError("Collection has not completed the current retrieval contract")
    manifest = root / "samples.parquet"
    digest = file_hash(manifest)
    stats = json.loads(statistics.read_text())
    if any(item.get("manifest_sha256") != digest for item in (marker, stats)):
        raise ValueError("Collection or statistics manifest mismatch")
    table = pd.read_parquet(manifest)
    if len(table) != marker["samples"] or not table.sample_id.is_unique:
        raise ValueError("Incomplete or duplicated samples")
    if contract := stats.get("data_contract"):
        packs = [
            {k: item[k] for k in ("path", "variables", "max_uncertainty")}
            for item in contract.get("label_packs", [])
        ]
        dataset = RetrievalDataset(
            manifest,
            "train",
            data_root=root,
            label_specs=contract.get("label_specs"),
            label_packs=packs,
        )
        if training_fingerprint(table, dataset.data_contract) != stats["training_fingerprint"]:
            raise ValueError("Dataset or label packs differ from frozen target statistics")
    return dict(samples=len(table), manifest_sha256=digest, version=VERSION)


def prepare(root, statistics, run_root, *, num_workers=4, config=None):
    """Audit, freeze training moments, and check both configs before marking ready."""
    root, statistics, run_root = (Path(p).resolve() for p in (root, statistics, run_root))
    os.environ.update(
        HYDROSTATE_DATA_ROOT=str(root),
        HYDROSTATE_MANIFEST=str(root / "samples.parquet"),
        HYDROSTATE_TARGET_STATS=str(statistics),
    )
    repo = Path(__file__).resolve().parents[3]
    provenance = run_root / "provenance"
    provenance.mkdir(parents=True, exist_ok=True)
    base = None
    for name, rank in (("mtlora", 4), ("mtlora_shared", 0)):
        cfg = Config.fromfile(str(repo / f"configs/hydrostate_{name}.py"))
        if config is not None:
            cfg = Config.fromfile(str(config))
            cfg.model.encoder.private_rank = rank
        configure_data_contract(cfg)
        if base is None:
            report = audit(
                root, label_packs=cfg.get("label_packs"), label_specs=cfg.get("label_specs")
            )
            print("Collection audit:", report, flush=True)
        if not statistics.exists():
            moments = compute_retrieval_statistics(cfg, num_workers=num_workers)
            statistics.parent.mkdir(parents=True, exist_ok=True)
            with statistics.open("x") as handle:
                handle.write(json.dumps(moments, indent=2) + "\n")
        configure_target_statistics(cfg, training=True)
        if cfg.model.encoder.private_rank != rank or cfg.model.encoder.shared_rank != 8:
            raise ValueError("Unexpected ranks in the paired model configurations")
        if base is not None:
            if cfg.train_dataloader.dataset != base.train_dataloader.dataset:
                raise ValueError("Paired models must use the same data")
        else:
            for key in ("train_dataloader", "val_dataloader", "test_dataloader"):
                dataset = RetrievalDataset(
                    **{k: v for k, v in cfg[key].dataset.items() if k != "type"}
                )
                for index in range(min(3, len(dataset))):
                    sample = dataset[index]
                    if sample["inputs"]["sensor_valid"].shape != (TIME_STEPS, len(SENSORS)) or set(
                        sample["inputs"]["target_time"]
                    ) != set(cfg.model.tasks):
                        raise ValueError("Invalid temporal inputs in prepared release")
        cfg.work_dir = str(run_root / ("lora" if rank == 0 else "mtlora"))
        cfg.dump(str(provenance / f"{name}.py"))
        base = cfg
    ready = dict(
        **verify(root, statistics),
        statistics_sha256=file_hash(statistics),
        temporal_sampling=VERSION,
    )
    atomic_json(provenance / "ready.json", ready)
    return ready


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "verify"))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--config", type=Path, help="Training config declaring optional label packs"
    )
    args = parser.parse_args()
    if args.num_workers < 0:
        parser.error("--num-workers must be nonnegative")
    if args.action == "prepare":
        if args.run_root is None:
            parser.error("prepare requires --run-root")
        result = prepare(
            args.data_root,
            args.statistics,
            args.run_root,
            num_workers=args.num_workers,
            config=args.config,
        )
    else:
        result = verify(args.data_root, args.statistics)
    print(json.dumps(result, indent=2))
