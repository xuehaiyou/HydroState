"""Frozen experiment inference and comparison on independent cover points."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope

from hydrostate.cli import configure_pretrained_encoder, configure_target_statistics
from hydrostate.data.cover_points.contract import CLASSES
from hydrostate.data.cover_points.overlap import training_overlap
from hydrostate.data.cover_points.prepare import atomic_parquet
from hydrostate.data.gee.contracts import atomic_json, fingerprint
from hydrostate.data.retrieval.contract import PROFILE
from hydrostate.datasets.cover_point_dataset import CoverPointDataset
from hydrostate.inference.regional import predict_records
from hydrostate.registry import MODELS
from hydrostate.storage.packs import file_hash

from .cover_point_metrics import summarize


def evaluate(root, config, checkpoint, output, *, batch_size=4, device="cuda", limit=None):
    if batch_size < 1 or (limit is not None and limit < 1):
        raise ValueError("Positive batch size/window limit required")
    root, output = Path(root), Path(output)
    if output.exists():
        raise FileExistsError(output)
    from mmengine.utils import import_modules_from_strings

    import hydrostate.models  # noqa: F401

    cfg = Config.fromfile(config)
    if cfg.get("custom_imports"):
        import_modules_from_strings(**cfg.custom_imports)
    init_default_scope("hydrostate")
    configure_pretrained_encoder(cfg)
    configure_target_statistics(cfg, training=False)
    if cfg.model.encoder.input_profile != PROFILE:
        raise ValueError("Checkpoint inputs differ from cover collection")
    published = root / "_PUBLISHED.json"
    if not published.exists():
        if not limit:
            raise ValueError("Full inference requires a published dataset")
        published = root / "audit" / "pilot.json"
        if not json.loads(published.read_text()).get("passed"):
            raise ValueError("Smoke inference requires a passed real-data pilot")
    dataset = CoverPointDataset(root, require_published=limit is None)
    points = pd.read_parquet(root / "points.parquet")
    plan = pd.read_parquet(root / "plan.parquet")
    overlaps, overlap_report = training_overlap(cfg, plan)
    torch.set_num_threads(1)
    model = MODELS.build(cfg.model)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"], strict=True)
    model.eval().to(device)
    if not set(CLASSES) <= set(model.tasks):
        raise ValueError("Checkpoint does not predict all three cover classes")
    output.mkdir(parents=True)
    atomic_parquet(overlaps, output / "training_overlap.parquet")
    provenance = dict(
        config=str(config),
        config_sha256=file_hash(config),
        checkpoint=str(checkpoint),
        checkpoint_sha256=file_hash(checkpoint),
        dataset_publication_sha256=file_hash(published),
        point_manifest_sha256=file_hash(root / "points.parquet"),
        model_config=cfg.model.to_dict(),
        training_dataset_config=cfg.train_dataloader.dataset.to_dict(),
        training_statistics_identity=fingerprint(cfg.model.target_statistics),
        overlap=overlap_report,
        scope="smoke" if limit else "full",
    )
    atomic_json(output / "provenance.json", provenance)
    n = min(len(dataset), limit) if limit else len(dataset)
    predictions = []
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    for start in range(0, n, batch_size):
        items = [dataset[i] for i in range(start, min(start + batch_size, n))]
        outputs = predict_records(model, [dict(inputs=item["inputs"]) for item in items], device)
        for item, predicted in zip(items, outputs, strict=True):
            labels = item["points"]
            rc = labels["cover_point_rc"]
            values = np.stack([predicted[c][0].numpy() for c in CLASSES])[:, rc[:, 0], rc[:, 1]].T
            support = predicted["prediction_valid"][0].numpy()[rc[:, 0], rc[:, 1]]
            finite = np.isfinite(values).all(axis=1)
            if np.any(support & ~finite):
                raise ValueError("Nonfinite cover prediction on valid support")
            if np.any(
                support & ((np.abs(values.sum(axis=1) - 1) > 0.01) | (values < -1e-6).any(axis=1))
            ):
                raise ValueError("Cover fractions do not close")
            for j, point_index in enumerate(labels["cover_point_index"]):
                predictions.append(
                    dict(
                        point_index=int(point_index),
                        label_valid=bool(labels["cover_point_valid"][j]),
                        prediction_valid=bool(support[j] and finite[j]),
                        predicted_class=int(values[j].argmax()) if finite[j] else -1,
                        **{c: float(values[j, k]) for k, c in enumerate(CLASSES)},
                    )
                )
        print(f"Predicted {min(start + batch_size, n)}/{n} windows", flush=True)
    if not predictions:
        raise ValueError("No predictions")
    # Preserve annotations from rejected input windows in the denominator report.
    frame = points.merge(
        pd.DataFrame(predictions), on="point_index", how="left", validate="one_to_one"
    )
    frame["label_valid"] = frame.label_valid.fillna(True).astype(bool)
    frame["prediction_valid"] = frame.prediction_valid.fillna(False).astype(bool)
    frame["predicted_class"] = frame.predicted_class.fillna(-1).astype(int)
    frame = frame.merge(overlaps, on="sample_id", validate="many_to_one")
    atomic_parquet(frame, output / "predictions.parquet")
    report = summarize(frame)
    report["scope"] = provenance["scope"]
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
        report["peak_cuda_memory_bytes"] = torch.cuda.max_memory_allocated()
    report["inference_seconds"] = time.monotonic() - started
    report["windows"] = n
    report["batch_size"] = batch_size
    atomic_json(output / "metrics.json", report)
    atomic_json(output / "_SUCCESS.json", dict(complete=True, scope=provenance["scope"]))
    return report


def compare(predictions, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    tables, identities, provenance = {}, set(), {}
    for entry in predictions:
        name, folder = entry.split("=", 1)
        folder = Path(folder)
        if name in tables:
            raise ValueError("Duplicate model name")
        report = json.loads((folder / "provenance.json").read_text())
        success = json.loads((folder / "_SUCCESS.json").read_text())
        if report["scope"] != "full" or not success["complete"]:
            raise ValueError("Comparison requires completed full inference")
        identities.add(report["dataset_publication_sha256"])
        provenance[name] = report
        table = pd.read_parquet(folder / "predictions.parquet").set_index("point_index", drop=False)
        if not table.index.is_unique:
            raise ValueError("Repeated point in predictions")
        tables[name] = table
    if len(identities) != 1 or not tables:
        raise ValueError("Models must use the same published dataset")
    common = None
    for frame in tables.values():
        keys = set(
            frame.index[frame.label_valid & frame.prediction_valid & ~frame.training_overlap]
        )
        common = keys if common is None else common & keys
    if not common:
        raise ValueError("No common independent valid points")
    report = dict(
        common_points=len(common), point_indices=sorted(common), models={}, provenance=provenance
    )
    for name, frame in tables.items():
        report["models"][name] = summarize(frame.loc[sorted(common)])
    atomic_json(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("predict")
    for name in ("root", "config", "checkpoint", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int)
    p = sub.add_parser("compare")
    p.add_argument("--predictions", action="append", required=True, help="NAME=RESULT_DIRECTORY")
    p.add_argument("--output", required=True)
    args = vars(parser.parse_args())
    command = args.pop("command")
    result = evaluate(**args) if command == "predict" else compare(**args)
    print(json.dumps(result, allow_nan=False))
