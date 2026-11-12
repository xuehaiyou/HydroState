"""Frozen-checkpoint station predictions on an externally fixed comparison manifest."""

import argparse
import json
import time
from pathlib import Path

import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope
from torch.utils.data import DataLoader

from hydrostate.cli import configure_pretrained_encoder, configure_target_statistics
from hydrostate.data.gee.contracts import atomic_json
from hydrostate.data.site.archives import sha256
from hydrostate.datasets.collate import hydro_collate
from hydrostate.registry import DATASETS, MODELS

from .site_baseline_scoring import common_comparison


def predict(config, checkpoint, pairs, output, batch_size=4):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    rows = [json.loads(line) for line in Path(pairs).read_text().splitlines()]
    cfg = Config.fromfile(config)
    init_default_scope("hydrostate")
    configure_pretrained_encoder(cfg)
    configure_target_statistics(cfg, training=False)
    dataset = DATASETS.build(dict(cfg.val_dataloader.dataset))
    wanted = {}
    for row in rows:
        if row.get("eligible"):
            wanted.setdefault((row["cohort"], row["sample_id"]), []).append(row)
    indices, ordered = [], []
    for cohort, site in enumerate(dataset.sites):
        for i, row in site.table.iterrows():
            key = (cohort, row.sample_id)
            if key in wanted:
                indices.append(dataset.offsets[cohort + 1] + i)
                ordered.append(wanted.pop(key))
    if wanted:
        raise ValueError("Comparison rows are not in the model validation split")
    if not indices:
        raise ValueError("No product-eligible station samples")
    torch.set_num_threads(1)
    model = MODELS.build(cfg.model)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"], strict=True)
    model.cuda().eval()
    loader = DataLoader(dataset, batch_size=batch_size, sampler=indices, num_workers=2,
                        pin_memory=True, collate_fn=hydro_collate)
    predictions, index = [], 0
    started = time.monotonic()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for batch in loader:
            for sample in model.val_step(batch):
                for row in ordered[index]:
                    task, j = f"site_{row['task']}", row["observation_index"]
                    predictions.append(dict(
                        key=row["key"], prediction=float(sample[f"supervised_{task}"][j]),
                        valid=bool(sample[f"valid_{task}"][j]),
                        original_target=float(sample[f"target_{task}"][j]),
                    ))
                index += 1
            if index % 100 == 0:
                print(f"Predicted {index}/{len(indices)} windows", flush=True)
    atomic_json(output, dict(config=str(config), checkpoint=str(checkpoint),
                            checkpoint_sha256=sha256(checkpoint), pairs_sha256=sha256(pairs),
                            windows=index, seconds=time.monotonic() - started, rows=predictions))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("predict")
    for name in ("config", "checkpoint", "pairs", "output"):
        command.add_argument("--" + name, required=True)
    command.add_argument("--batch-size", type=int, default=4)
    command = sub.add_parser("score")
    command.add_argument("--pairs", required=True)
    command.add_argument("--prediction", action="append", required=True, help="NAME=JSON")
    command.add_argument("--output", required=True)
    args = vars(parser.parse_args())
    if args.pop("command") == "predict":
        predict(**args)
        return
    pairs = Path(args["pairs"])
    predictions, provenance = {}, {}
    for value in args["prediction"]:
        name, path = value.split("=", 1)
        if name in predictions or name == "product":
            raise ValueError("Unique model names are required")
        report = json.loads(Path(path).read_text())
        if report["pairs_sha256"] != sha256(pairs):
            raise ValueError("Model predictions were generated from another pairing manifest")
        predictions[name] = report.pop("rows")
        provenance[name] = report
    report, selected = common_comparison(
        [json.loads(line) for line in pairs.read_text().splitlines()], predictions)
    atomic_json(Path(args["output"]), dict(metrics=report, models=provenance,
                pairs_sha256=sha256(pairs), common_keys=[r["key"] for r in selected],
                scope="validation intersection; SM common morning dates estimate the 8day state; "
                "ET local/UTC timing offset and native spatial resolution remain limitations"))
