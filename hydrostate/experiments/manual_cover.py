"""Freeze the three approved architectures on jointly isolated human supervision."""

import copy
import json
from pathlib import Path

import pandas as pd
from mmengine.config import Config
from mmengine.registry import init_default_scope

from hydrostate.data.cover_points.prepare import atomic_parquet
from hydrostate.data.gee.contracts import atomic_json
from hydrostate.data.manual_cover.splits import MARGIN_M
from hydrostate.data.retrieval.statistics import compute_retrieval_statistics
from hydrostate.data.site.splits import CircleIndex, smap_circle, tile_circle
from hydrostate.data.site.statistics import cohort_statistics
from hydrostate.datasets.joint_dataset import JointRetrievalDataset
from hydrostate.storage.packs import file_hash

VARIANTS = ("CGC", "CoSA_Router", "B")
RESIDUAL_VARIANTS = ("bg_residual", "bg_residual_centered")


def prepare_residual_pair(reference, root, output):
    """Reuse the frozen joint selection, statistics and budget for both residual models."""
    reference, root, output = Path(reference), Path(root), Path(output)
    if output.resolve() == reference.resolve() or reference.resolve() in output.resolve().parents:
        raise ValueError("Residual experiments need their own output outside the reference run")
    config = reference / "CGC" / "config.py"
    frozen = json.loads((reference / "prepared.json").read_text())
    publication = file_hash(root / "_PUBLISHED.json")
    if frozen["publication_sha256"] != publication:
        raise ValueError("Reference selection belongs to a different manual Cover publication")
    cfg = Config.fromfile(config)
    for loader in (cfg.train_dataloader, cfg.val_dataloader, cfg.test_dataloader):
        if loader.dataset.type != "ManualJointDataset" or Path(loader.dataset.manual_root) != root:
            raise ValueError("Expected joint manual Cover loaders from this release")
    if (
        cfg.model.router.type != "CGCRouter"
        or cfg.default_hooks.checkpoint.save_best != "manual_joint_score"
    ):
        raise ValueError("Expected manual joint CGC reference config")
    if cfg.load_from is not None or cfg.resume:
        raise ValueError("Use the reference architecture config from before training")
    report = dict(
        variants=list(RESIDUAL_VARIANTS),
        publication_sha256=publication,
        reference_config=str(config),
        reference_config_sha256=file_hash(config),
    )
    receipt = output / "prepared.json"
    if receipt.exists():
        if json.loads(receipt.read_text()) != report:
            raise ValueError("The residual experiment reference has changed")
        return
    for name in RESIDUAL_VARIANTS:
        directory = output / name
        directory.mkdir(parents=True, exist_ok=True)
        cfg.model.type = "BackgroundResidualModel"
        cfg.model.router = dict(center=name == "bg_residual_centered")
        cfg.work_dir = str(directory / "work")
        cfg.dump(str(directory / "config.py"))
    atomic_json(receipt, report)


def supports(dataset):
    rows = []
    for part in dataset.components:
        for row in part.table.to_dict("records"):
            common = dict(sample_id=row["sample_id"])
            rows.append(dict(tile_circle(row), **common))
            counts = row.get("valid_label_counts")
            if isinstance(counts, dict) and counts.get("soil_moisture_valid", 0):
                attrs = json.loads((part.root / row["label_path"] / ".zattrs").read_text())
                rows.append(dict(smap_circle(attrs["sm_quality"]["support"]), **common))
    return rows


def separated(table, index):
    keep, removed = [], []
    for row in table.to_dict("records"):
        circle = tile_circle(row)
        hits = index.overlaps(circle["lon"], circle["lat"], circle["radius_m"] + MARGIN_M)
        (removed if hits else keep).append(row["sample_id"])
    return table[table.sample_id.isin(keep)].reset_index(drop=True), removed


def prepare(models, manual_root, output):
    init_default_scope("hydrostate")
    import hydrostate.models  # noqa: F401

    root, output = Path(manual_root), Path(output)
    if (output / "prepared.json").exists():
        raise FileExistsError("Training preparation already frozen")
    output.mkdir(parents=True, exist_ok=True)
    definitions = {m["name"]: m for m in json.loads(Path(models).read_text())}
    base = Config.fromfile(definitions["B"]["config"])
    cfg_definition = dict(base.train_dataloader.dataset)
    cfg_definition.pop("type")
    source = JointRetrievalDataset(**cfg_definition)
    original_selection = json.loads(
        (Path(cfg_definition["selection_root"]) / "audit.json").read_text()
    )
    selection = output / "selection"
    selection.mkdir(exist_ok=True)
    exclusions = pd.read_parquet(root / "training_exclusions.parquet")
    excluded_ids = set(exclusions.sample_id)
    human = pd.read_parquet(root / "samples.parquet")
    manual_test = CircleIndex(
        [tile_circle(r) for r in human[human.split.eq("test")].to_dict("records")]
    )
    old_holdouts = []
    datasets = {}
    for split in ("train", "validation", "test"):
        options = dict(cfg_definition, split=split, training=False)
        if split == "test":
            options["selection_root"] = None
        data = JointRetrievalDataset(**options)
        datasets[split] = data
        if split != "train":
            # Isolation against the full published old holdouts, not just the search subset.
            complete = JointRetrievalDataset(**dict(options, selection_root=None))
            old_holdouts.extend(supports(complete))
    audit = copy.deepcopy(original_selection)
    audit["splits"] = {}
    removals = []
    for split, dataset in datasets.items():
        fraction = 0.1 if split == "train" else 0.2
        audit["splits"][split] = {}
        for i, part in enumerate(dataset.components):
            table = part.table.copy()
            if split == "train":
                removed = table[table.sample_id.isin(excluded_ids)].sample_id.tolist()
                table = table[~table.sample_id.isin(excluded_ids)].reset_index(drop=True)
            elif split == "validation":
                table, removed = separated(table, manual_test)
            else:
                removed = []
            if table.empty:
                raise ValueError(f"Isolation emptied {split} component {i}")
            path = selection / f"{split}_{i}.parquet"
            atomic_parquet(table, path)
            audit["splits"][split][str(i)] = dict(
                subsets={str(fraction): dict(manifest=path.name, sha256=file_hash(path))}
            )
            removals.append(dict(split=split, component=i, excluded_sample_ids=removed))
    atomic_json(selection / "audit.json", audit)
    # Manual train never approaches original validation/test windows or SMAP supports.
    retained, manual_removed = separated(human[human.split.eq("train")], CircleIndex(old_holdouts))
    # Manual validation must also be isolated from old test regions.
    old_test = supports(datasets["test"])
    manual_val, val_removed = separated(human[human.split.eq("val")], CircleIndex(old_test))
    retained = pd.concat([retained, manual_val, human[human.split.eq("test")]], ignore_index=True)
    for kind in ("glacier", "sen1floods11"):
        if set(retained[retained.source.eq(kind)].split) != {"train", "val", "test"}:
            raise ValueError(f"Isolation emptied {kind} split; revise region allocation")
    manual_path = output / "manual_selection.parquet"
    atomic_parquet(retained[["sample_id", "source", "split", "region"]], manual_path)
    atomic_json(
        manual_path.with_suffix(".json"),
        dict(sha256=file_hash(manual_path), publication_sha256=file_hash(root / "_PUBLISHED.json")),
    )
    atomic_json(
        output / "isolation.json",
        dict(
            original_removed=removals,
            manual_train_removed=manual_removed,
            manual_validation_removed=val_removed,
            margin_m=MARGIN_M,
            rules="all_dates_cross_source_train_val_test_isolation",
        ),
    )
    # Recompute only from the actual retained product/site training records.
    stats_cfg = copy.deepcopy(base)
    product_path = selection / "train_0.parquet"
    stats_cfg.train_dataloader.dataset = dict(
        type="RetrievalDataset",
        manifest=str(product_path),
        data_root=str(source.product.root),
        split="train",
        training=False,
        tasks=base.tasks,
        input_profile=base.input_profile,
    )
    product_stats = compute_retrieval_statistics(stats_cfg, num_workers=0, batch_size=4)
    atomic_json(output / "product_statistics.json", product_stats)
    site_stats = cohort_statistics(
        [
            dict(manifest=str(selection / f"train_{i}.parquet"), data_root=str(site.root))
            for i, site in enumerate(source.sites, 1)
        ]
    )
    atomic_json(output / "site_statistics.json", site_stats)
    for name in VARIANTS:
        cfg = Config.fromfile(definitions[name]["config"])
        directory = output / name
        directory.mkdir(exist_ok=True)
        cfg.custom_imports = dict(
            imports=[
                "hydrostate.models",
                "hydrostate.datasets",
                "hydrostate.evaluation",
                "hydrostate.datasets.manual_joint_dataset",
                "hydrostate.evaluation.manual_joint_metrics",
            ],
            allow_failed_imports=False,
        )
        cfg.load_from, cfg.resume = None, False
        cfg.model.manual_cover_weight = 2.0
        cfg.model.target_statistics = str(output / "product_statistics.json")
        cfg.model.site_supervision.statistics = str(output / "site_statistics.json")
        cfg.train_cfg.max_epochs, cfg.train_cfg.val_interval = 6, 1
        cfg.optim_wrapper.accumulative_counts = 2
        cfg.optim_wrapper.optimizer.lr = 1e-4
        for key, split in [
            ("train_dataloader", "train"),
            ("val_dataloader", "validation"),
            ("test_dataloader", "test"),
        ]:
            loader = copy.deepcopy(cfg.train_dataloader if split == "train" else cfg.val_dataloader)
            loader.batch_size = 5 if split == "train" else 8
            loader.num_workers = 2
            loader.dataset.update(
                type="ManualJointDataset",
                split=split,
                training=split == "train",
                selection_root=str(selection),
                manual_root=str(root),
                manual_selection=str(manual_path),
                train_fraction=0.1,
                validation_fraction=0.2,
            )
            loader.sampler.shuffle = split == "train"
            cfg[key] = loader
        cfg.default_hooks.checkpoint.save_best = "manual_joint_score"
        cfg.default_hooks.checkpoint.rule = "greater"
        cfg.val_evaluator.type = "ManualJointMetric"
        cfg.val_evaluator.selection = "site"
        cfg.val_evaluator.target_statistics = str(output / "product_statistics.json")
        cfg.val_evaluator.site_statistics = str(output / "site_statistics.json")
        cfg.test_evaluator = copy.deepcopy(cfg.val_evaluator)
        cfg.work_dir = str(directory / "work")
        # Fixed effective updates: 100 warmup updates, cosine over the six epochs.
        cfg.param_scheduler = [
            dict(type="LinearLR", start_factor=0.01, by_epoch=False, begin=0, end=200),
            dict(
                type="CosineAnnealingLR", by_epoch=True, begin=0, end=6, T_max=6, eta_min_ratio=0.01
            ),
        ]
        cfg.dump(str(directory / "config.py"))
    report = dict(
        variants=list(VARIANTS),
        manual_windows=retained.groupby(["source", "split"])
        .size()
        .reset_index(name="windows")
        .to_dict("records"),
        product_training_windows=len(pd.read_parquet(product_path)),
        publication_sha256=file_hash(root / "_PUBLISHED.json"),
        initialization="pretrained_backbone_no_old_task_checkpoint",
    )
    atomic_json(output / "prepared.json", report)
    return report
