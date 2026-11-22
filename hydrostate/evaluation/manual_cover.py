"""Source-separated fraction and classification scores on immutable spatial holdouts."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import zarr
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.utils import import_modules_from_strings
from numcodecs import Blosc

from hydrostate.cli import configure_pretrained_encoder, configure_target_statistics
from hydrostate.data.cover_points.overlap import training_overlap
from hydrostate.data.cover_points.prepare import atomic_parquet
from hydrostate.data.gee.contracts import atomic_json
from hydrostate.data.manual_cover.contract import CLASSES
from hydrostate.data.retrieval.contract import PROFILE
from hydrostate.datasets.manual_cover_dataset import ManualCoverDataset
from hydrostate.inference.regional import predict_records
from hydrostate.registry import MODELS
from hydrostate.storage.packs import file_hash


class Scores:
    def __init__(self, source):
        self.source = source
        self.sums = {
            c: np.zeros(4, dtype="float64")
            for c in (CLASSES if source == "glacier" else ("water",))
        }
        self.confusion = np.zeros((3, 3) if source == "glacier" else (2, 2), dtype="int64")
        self.windows = 0

    def add(self, predicted, labels, support):
        self.windows += 1
        for i, c in enumerate(CLASSES):
            if c not in self.sums:
                continue
            mask = support & labels[c + "_valid"][0].astype(bool)
            error = predicted[i][mask].astype("float64") - labels[c][0][mask]
            self.sums[c] += [len(error), np.abs(error).sum(), (error**2).sum(), error.sum()]
        mask = support & labels["water_valid"][0].astype(bool)
        if self.source == "glacier":
            truth = np.stack([labels[c][0] for c in CLASSES]).argmax(axis=0)[mask]
            estimate = predicted.argmax(axis=0)[mask]
        else:
            truth = (labels["water"][0][mask] >= 0.5).astype(int)
            estimate = (predicted[0][mask] >= 0.5).astype(int)
        np.add.at(self.confusion, (truth, estimate), 1)

    def report(self):
        fractions = {}
        for c, (n, absolute, squared, signed) in self.sums.items():
            fractions[c] = dict(
                count=int(n),
                mae=absolute / n if n else None,
                rmse=float(np.sqrt(squared / n)) if n else None,
                bias=signed / n if n else None,
            )
        classes = {}
        for i, name in enumerate(CLASSES) if self.source == "glacier" else [(1, "water")]:
            tp = int(self.confusion[i, i])
            actual, estimated = int(self.confusion[i].sum()), int(self.confusion[:, i].sum())
            classes[name] = dict(
                support=actual,
                predicted_count=estimated,
                precision=tp / estimated if estimated and actual else None,
                recall=tp / actual if actual else None,
                f1=2 * tp / (actual + estimated) if actual else None,
                iou=tp / (actual + estimated - tp) if actual else None,
            )
        total = int(self.confusion.sum())
        return dict(
            windows=self.windows,
            fraction_metrics=fractions,
            classes=classes,
            confusion_matrix=self.confusion.tolist(),
            class_order=list(CLASSES) if self.source == "glacier" else ["nonwater", "water"],
            accuracy=float(np.trace(self.confusion) / total)
            if total and self.source == "glacier"
            else None,
            interpretation="pure_30m_pixel_assumption"
            if self.source == "glacier"
            else "manual_raster_area_fraction",
            water_threshold=0.5,
        )


def evaluate(root, config, checkpoint, output, batch_size=16, limit=None):
    if not 1 <= batch_size <= 16:
        raise ValueError("This evaluation uses a conservative batch cap of 16")
    output, root = Path(output), Path(root)
    if output.exists():
        raise FileExistsError(output)
    dataset = ManualCoverDataset(root, require_published=limit is None)
    if limit and not json.loads((root / "audit/pilot.json").read_text())["passed"]:
        raise ValueError("Real input pilot must pass")
    dataset.table = dataset.table[dataset.table.split.isin(["val", "test"])].reset_index(drop=True)
    if limit:
        dataset.table = dataset.table.head(limit)
    if dataset.table.empty:
        raise ValueError("No evaluation windows")
    cfg = Config.fromfile(config)
    import hydrostate.models  # noqa: F401

    if cfg.get("custom_imports"):
        import_modules_from_strings(**cfg.custom_imports)
    init_default_scope("hydrostate")
    configure_pretrained_encoder(cfg)
    configure_target_statistics(cfg, training=False)
    if cfg.model.encoder.input_profile != PROFILE:
        raise ValueError("Input profile mismatch")
    overlaps, overlap_report = training_overlap(cfg, dataset.table)
    overlap = overlaps.set_index("sample_id").training_overlap.to_dict()
    torch.set_num_threads(1)
    model = MODELS.build(cfg.model)
    model.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"], strict=True
    )
    model.eval().cuda()
    output.mkdir(parents=True)
    atomic_parquet(dataset.table, output / "samples.parquet")
    atomic_parquet(overlaps, output / "training_overlap.parquet")
    atomic_json(
        output / "provenance.json",
        dict(
            config=str(config),
            config_sha256=file_hash(config),
            checkpoint=str(checkpoint),
            checkpoint_sha256=file_hash(checkpoint),
            dataset_identity=file_hash(
                root / ("_PUBLISHED.json" if not limit else "audit/pilot.json")
            ),
            batch_size=batch_size,
            overlap=overlap_report,
            scope="smoke" if limit else "full",
        ),
    )
    z = zarr.open_group(str(output / "predictions.zarr"), mode="w")
    n = len(dataset)
    pred = z.create_dataset(
        "fractions",
        shape=(n, 3, 40, 40),
        chunks=(1, 3, 40, 40),
        dtype="float32",
        compressor=Blosc(cname="zstd", clevel=3),
    )
    masks = z.create_dataset("valid", shape=(n, 40, 40), chunks=(1, 40, 40), dtype="bool")
    scores = {}
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    for start in range(0, n, batch_size):
        items = [dataset[i] for i in range(start, min(start + batch_size, n))]
        outputs = predict_records(model, [dict(inputs=x["inputs"]) for x in items], "cuda")
        for i, (item, values) in enumerate(zip(items, outputs, strict=True), start):
            row = dataset.table.iloc[i]
            p = np.stack([values[c][0].numpy() for c in CLASSES])
            valid = values["prediction_valid"][0].numpy().astype(bool)
            if not np.isfinite(p[:, valid]).all() or not np.allclose(
                p[:, valid].sum(axis=0), 1, atol=0.01
            ):
                raise ValueError("Invalid predicted fractions")
            pred[i], masks[i] = p, valid
            labels = {k: v.numpy() for k, v in item["data_samples"].items() if k != "metainfo"}
            for group in (f"{row.source}/{row.split}", f"{row.source}/{row.split}/{row.region}"):
                for scope in ("all_valid", "outside_checkpoint_training_proximity"):
                    key = group + "/" + scope
                    s = scores.setdefault(key, Scores(row.source))
                    if scope == "all_valid" or not overlap[row.sample_id]:
                        s.add(p, labels, valid)
        print(f"Predicted {min(start + batch_size, n)}/{n}", flush=True)
    report = dict(
        scope="smoke" if limit else "full",
        windows=n,
        batch_size=batch_size,
        inference_seconds=time.monotonic() - started,
        peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        groups={k: s.report() for k, s in scores.items()},
    )
    atomic_json(output / "metrics.json", report)
    atomic_json(output / "_SUCCESS.json", dict(complete=True, scope=report["scope"]))
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "config", "checkpoint", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--limit", type=int)
    result = evaluate(**vars(p.parse_args()))
    print(json.dumps({k: v for k, v in result.items() if k != "groups"}))
