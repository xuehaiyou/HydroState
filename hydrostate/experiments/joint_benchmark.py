"""Disposable joint-training batch probe including real Zarr reads and validation."""

import argparse
import json
import time
from functools import partial
from pathlib import Path

import torch
from mmengine.config import Config
from mmengine.optim import build_optim_wrapper
from mmengine.registry import init_default_scope
from mmengine.runner import set_random_seed
from torch.utils.data import DataLoader

from hydrostate.cli import configure_pretrained_encoder, configure_target_statistics
from hydrostate.data.gee.contracts import atomic_json
from hydrostate.data.site.training_smoke import representative_indices
from hydrostate.datasets.collate import hydro_collate
from hydrostate.experiments.gpu_memory import CudaMemoryMeter
from hydrostate.models.olmoearth_encoder import _broadcast_sdpa
from hydrostate.registry import DATASETS, MODELS


def benchmark(config, output, *, batch_size, steps=8, workers=2):
    """Measure one candidate in its own process; OOM never leaves a training checkpoint."""
    if batch_size < 1 or steps < 3 or workers < 0:
        raise ValueError("Require positive batch, at least three steps and nonnegative workers")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one CUDA GPU is required")
    started = time.monotonic()
    cfg = Config.fromfile(config)
    init_default_scope("hydrostate")
    configure_pretrained_encoder(cfg)
    configure_target_statistics(cfg, training=True)
    set_random_seed(cfg.randomness.seed)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = cfg.env_cfg.cudnn_benchmark
    dataset = DATASETS.build(dict(cfg.train_dataloader.dataset))
    representative = representative_indices(dataset)
    generator = torch.Generator().manual_seed(cfg.randomness.seed)
    required = batch_size * steps
    if required > len(dataset):
        raise ValueError("Probe exceeds the unique training windows")
    chosen = list(dict.fromkeys(representative))
    selected = set(chosen)
    chosen.extend(i for i in torch.randperm(len(dataset), generator=generator).tolist()
                  if i not in selected)
    indices = chosen[:required]
    loader = DataLoader(dataset, batch_size=batch_size, sampler=indices, num_workers=workers,
                        pin_memory=True, collate_fn=hydro_collate)
    model = MODELS.build(cfg.model).cuda().train()
    cfg.optim_wrapper.accumulative_counts = 1
    optimizer = build_optim_wrapper(model, cfg.optim_wrapper)
    optimizer.initialize_count_status(model, 0, steps)
    report = dict(config=str(config), batch_size=batch_size, workers=workers, status="running",
                  gpu=torch.cuda.get_device_name(), accumulation=1,
                  gpu_total_gib=torch.cuda.get_device_properties(0).total_memory / 2**30,
                  steps=[], scope="Disposable train-split updates including input I/O; "
                  "not model accuracy or a reusable checkpoint", train_samples=len(dataset),
                  sample_ids=[dataset.product.table.iloc[i].sample_id if i < len(dataset.product)
                              else f"joint_index:{i}" for i in indices])
    attention = [block.attn for block in model.encoder.backbone.model.blocks]
    report["gradient_checkpointing"] = cfg.model.encoder.gradient_checkpointing
    report["broadcast_attention_blocks"] = sum(
        isinstance(layer.sdpa, partial) and layer.sdpa.func is _broadcast_sdpa
        for layer in attention)
    memory = CudaMemoryMeter()
    try:
        iterator = iter(loader)
        torch.cuda.reset_peak_memory_stats()
        for step in range(steps):
            torch.cuda.synchronize()
            began = time.monotonic()
            raw = next(iterator)
            loaded = time.monotonic()
            losses = model.train_step(raw, optimizer)
            if any(not torch.isfinite(value).all() for value in losses.values()
                   if torch.is_tensor(value)):
                raise RuntimeError("Nonfinite joint training loss")
            measured = memory.sample()
            report["steps"].append(dict(
                seconds=time.monotonic() - began,
                data_wait_seconds=loaded - began,
                **measured))
            print(json.dumps(dict(batch=batch_size, step=step, **report["steps"][-1])), flush=True)
        train_times = [row["seconds"] for row in report["steps"][2:]]
        report["train_seconds_per_sample"] = sum(train_times) / len(train_times) / batch_size
        report["train_samples_per_second"] = 1 / report["train_seconds_per_sample"]
        model.eval()
        val = DATASETS.build(dict(cfg.val_dataloader.dataset))
        report["validation_samples"] = len(val)
        chosen = representative_indices(val)
        chosen = [chosen[i % len(chosen)] for i in range(batch_size * 3)]
        validation = DataLoader(val, batch_size=batch_size, sampler=chosen, num_workers=workers,
                                pin_memory=True, collate_fn=hydro_collate)
        iterator = iter(validation)
        times = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(3):
                torch.cuda.synchronize()
                began = time.monotonic()
                model.val_step(next(iterator))
                memory.sample()
                times.append(time.monotonic() - began)
        report["validation_seconds_per_sample"] = sum(times[1:]) / 2 / batch_size
        report["status"] = "passed"
    except torch.cuda.OutOfMemoryError as error:
        report.update(status="oom", error=str(error))
    report.update(**memory.sample(), elapsed_seconds=time.monotonic() - started)
    atomic_json(output, report)
    print(json.dumps({k: v for k, v in report.items() if k not in ("sample_ids", "steps")}),
          flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hydrostate_joint.py")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", required=True, type=int)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    benchmark(args.config, args.output, batch_size=args.batch_size, steps=args.steps,
              workers=args.workers)
