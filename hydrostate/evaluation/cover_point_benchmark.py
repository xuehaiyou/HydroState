"""Measure safe single-GPU inference batches on real cover-window inputs."""

import argparse
import gc
import json
import time
from pathlib import Path

import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.utils import import_modules_from_strings

from hydrostate.cli import configure_pretrained_encoder, configure_target_statistics
from hydrostate.data.gee.contracts import atomic_json
from hydrostate.datasets.cover_point_dataset import CoverPointDataset
from hydrostate.inference.regional import predict_records
from hydrostate.registry import MODELS


def benchmark(root, config, checkpoint, output):
    if Path(output).exists():
        raise FileExistsError(output)
    cfg = Config.fromfile(config)
    if cfg.get("custom_imports"):
        import_modules_from_strings(**cfg.custom_imports)
    init_default_scope("hydrostate")
    configure_pretrained_encoder(cfg)
    configure_target_statistics(cfg, training=False)
    dataset = CoverPointDataset(root, require_published=False)
    if not (Path(root) / "audit" / "pilot.json").exists():
        raise ValueError("Real pilot required")
    torch.set_num_threads(1)
    model = MODELS.build(cfg.model)
    model.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"], strict=True
    )
    model.eval().cuda()
    samples = [dict(inputs=dataset[i]["inputs"]) for i in range(min(len(dataset), 6))]
    total = torch.cuda.get_device_properties(0).total_memory
    results, selected = [], None
    for batch_size in (1, 2, 4, 8, 16, 32, 64):
        records = [samples[i % len(samples)] for i in range(batch_size)]
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            predict_records(model, records, "cuda")
            torch.cuda.synchronize()
            started = time.monotonic()
            for _ in range(2):
                predict_records(model, records, "cuda")
            torch.cuda.synchronize()
            seconds = (time.monotonic() - started) / 2
            peak = torch.cuda.max_memory_allocated()
            result = dict(
                batch_size=batch_size,
                peak_bytes=peak,
                seconds=seconds,
                windows_per_second=batch_size / seconds,
                safe=peak <= 0.85 * total,
            )
            results.append(result)
            print(json.dumps(result), flush=True)
            if result["safe"]:
                selected = batch_size
            else:
                break
        except torch.cuda.OutOfMemoryError:
            results.append(dict(batch_size=batch_size, status="cuda_oom"))
            gc.collect()
            torch.cuda.empty_cache()
            break
    if selected is None:
        raise RuntimeError("No batch retains 15% memory headroom")
    report = dict(
        selected_batch_size=selected,
        measurements=results,
        gpu=torch.cuda.get_device_name(0),
        total_memory_bytes=total,
        config=str(config),
        checkpoint=str(checkpoint),
        scope="repeated_real_pilot_windows_inference_only",
    )
    atomic_json(Path(output), report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "config", "checkpoint", "output"):
        parser.add_argument("--" + name, required=True)
    print(json.dumps(benchmark(**vars(parser.parse_args()))))
