"""Prepare and train a single frozen OlmoEarth BDTA on all isolated published windows."""

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
from mmengine.registry import init_default_scope

from hydrostate.data.cover_points.prepare import atomic_parquet
from hydrostate.data.gee.contracts import atomic_json
from hydrostate.data.manual_cover.splits import MARGIN_M
from hydrostate.data.retrieval.statistics import compute_retrieval_statistics
from hydrostate.data.site.splits import CircleIndex, smap_circle, tile_circle
from hydrostate.data.site.statistics import cohort_statistics
from hydrostate.datasets.joint_dataset import JointRetrievalDataset
from hydrostate.datasets.manual_joint_dataset import ManualJointDataset
from hydrostate.experiments.dense_adaptation import configure
from hydrostate.experiments.gpu_memory import has_memory_headroom
from hydrostate.experiments.manual_cover import separated
from hydrostate.experiments.manual_cover_launch import finish_test
from hydrostate.storage.packs import file_hash

EPOCHS = 50
BATCH = 10


def full_config(reference, directory):
    cfg = configure(reference, directory, "bdta")
    cfg.custom_imports.imports.extend(
        [
            "hydrostate.losses.fixed_mse",
            "hydrostate.models.cosa.fixed_bdta",
            "hydrostate.datasets.full_source_sampler",
        ]
    )
    cfg.model.type = "FixedSupervisionBDTA"
    cfg.model.loss = dict(type="FixedStandardizedMSE")
    cfg.model.task_loss_weights = {}
    cfg.model.manual_cover_weight = 2 / 9
    cfg.model.site_supervision.loss_weights = dict(site_sm=2 / 9, site_et=2 / 9)
    cfg.model.target_statistics = str(directory / "product_statistics.json")
    cfg.model.site_supervision.statistics = str(directory / "site_statistics.json")
    for loader in (cfg.train_dataloader, cfg.val_dataloader, cfg.test_dataloader):
        loader.dataset.update(
            selection_root=str(directory / "selection"), train_fraction=1.0, validation_fraction=1.0
        )
    for evaluator in (cfg.val_evaluator, cfg.test_evaluator):
        evaluator.target_statistics = cfg.model.target_statistics
        evaluator.site_statistics = cfg.model.site_supervision.statistics
    cfg.train_dataloader.update(
        batch_size=BATCH, drop_last=False, sampler=dict(type="FullSourceSampler", batch_size=BATCH)
    )
    cfg.optim_wrapper.accumulative_counts = 1
    cfg.train_cfg.max_epochs = EPOCHS
    cfg.param_scheduler[0].end = 500
    cfg.param_scheduler[1].update(end=EPOCHS, T_max=EPOCHS)
    cfg.custom_hooks = [
        *cfg.get("custom_hooks", []),
        dict(
            type="EarlyStoppingHook",
            monitor="manual_joint_score",
            rule="greater",
            min_delta=0.0001,
            patience=5,
            strict=True,
        ),
    ]
    return cfg


def prepare(reference, directory):
    cfg = full_config(reference, directory)
    source = dict(cfg.train_dataloader.dataset)
    root = Path(source.pop("manual_root"))
    manual = Path(source.pop("manual_selection"))
    source.pop("type")
    source.update(selection_root=None, training=False)
    # This manual selection was already isolated against the FULL legacy holdouts.
    pub = file_hash(root / "_PUBLISHED.json")
    binding = json.loads(manual.with_suffix(".json").read_text())
    if binding["publication_sha256"] != pub or binding["sha256"] != file_hash(manual):
        raise ValueError("Manual selection/publication mismatch")
    kept_manual = pd.read_parquet(manual)
    excluded = set(pd.read_parquet(root / "training_exclusions.parquet").sample_id)
    human = pd.read_parquet(root / "samples.parquet")
    manual_test = CircleIndex(
        [tile_circle(r) for r in human[human.split.eq("test")].to_dict("records")]
    )
    selection = directory / "selection"
    selection.mkdir(parents=True, exist_ok=True)
    audit = dict(version=1, splits={})
    summary = dict(manual_publication=pub, manual_selection_sha256=file_hash(manual), windows={})
    for split in ("train", "validation", "test"):
        ds = JointRetrievalDataset(**dict(source, split=split))
        if split == "train":
            training = ds
        components = {}
        sizes = []
        for i, part in enumerate(ds.components):
            table = part.table
            if split == "train":
                table = table[~table.sample_id.isin(excluded)]
            elif split == "validation":
                table, _ = separated(table, manual_test)
                if i == 0:
                    overlaps = []
                    for row in table.itertuples():
                        if not row.valid_label_counts.get("soil_moisture_valid", 0):
                            continue
                        attrs = json.loads((part.root / row.label_path / ".zattrs").read_text())
                        cell = smap_circle(attrs["sm_quality"]["support"])
                        if manual_test.overlaps(
                            cell["lon"], cell["lat"], cell["radius_m"] + MARGIN_M
                        ):
                            overlaps.append(row.sample_id)
                    table = table[~table.sample_id.isin(overlaps)]
            path = selection / f"{split}_{i}.parquet"
            atomic_parquet(table.reset_index(drop=True), path)
            components[str(i)] = dict(
                subsets={"1.0": dict(manifest=path.name, sha256=file_hash(path))}
            )
            sizes.append(len(table))
        audit["splits"][split] = components
        h = kept_manual[kept_manual.split.eq("val" if split == "validation" else split)]
        summary["windows"][split] = dict(
            components=sizes, manual=h.groupby("source").size().to_dict(), total=sum(sizes) + len(h)
        )
    atomic_json(selection / "audit.json", audit)
    stats_cfg = copy.deepcopy(cfg)
    stats_cfg.train_dataloader.dataset = dict(
        type="RetrievalDataset",
        manifest=str(selection / "train_0.parquet"),
        data_root=str(training.product.root),
        split="train",
        training=False,
        tasks=cfg.tasks,
        input_profile=cfg.input_profile,
    )
    atomic_json(
        directory / "product_statistics.json",
        compute_retrieval_statistics(stats_cfg, num_workers=3, batch_size=8),
    )
    atomic_json(
        directory / "site_statistics.json",
        cohort_statistics(
            [
                dict(manifest=str(selection / f"train_{i}.parquet"), data_root=str(site.root))
                for i, site in enumerate(training.sites, 1)
            ]
        ),
    )
    atomic_json(directory / "prepared.json", summary)


def train(reference, directory):
    cfg = full_config(reference, directory)
    counts = json.loads((directory / "prepared.json").read_text())["windows"]
    for name, split in (("train", "train"), ("val", "validation"), ("test", "test")):
        options = dict(cfg[name + "_dataloader"].dataset)
        options.pop("type")
        size = len(ManualJointDataset(**options))
        if size != counts[split]["total"]:
            raise ValueError(f"Frozen {split} size differs from installed release")
    config = directory / "train_config.py"
    cfg.dump(str(config))
    benchmark = directory / "benchmark.json"
    if not (Path(cfg.work_dir) / "last_checkpoint").exists():
        subprocess.run(
            [
                sys.executable,
                "tools/benchmarks/joint_training.py",
                "--config",
                str(config),
                "--output",
                str(benchmark),
                "--batch-size",
                str(BATCH),
                "--validation-batch",
                "8",
                "--steps",
                "6",
                "--workers",
                "2",
            ],
            check=True,
        )
        report = json.loads(benchmark.read_text())
        if report["status"] != "passed" or not has_memory_headroom(report):
            raise RuntimeError(f"GPU preflight failed: {benchmark}")
        options = []
    else:
        options = ["--cfg-options", "resume=True"]
    subprocess.run([sys.executable, "tools/train.py", str(config), *options], check=True)
    finish_test(str(config))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "train"))
    parser.add_argument("reference", type=Path)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    init_default_scope("hydrostate")
    (prepare if args.stage == "prepare" else train)(args.reference, args.directory)
