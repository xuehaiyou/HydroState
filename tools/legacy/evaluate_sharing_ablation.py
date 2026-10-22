#!/usr/bin/env python
"""Evaluate epoch 50 and validation-selected weights, without test-set selection."""

import argparse
import gc
import math
import time
from pathlib import Path

import torch
from mmengine.config import Config
from mmengine.runner import Runner

from hydrostate.cli import configure_target_statistics
from tools.legacy.submit_shared_private import best_checkpoint, save_record
from tools.legacy.submit_sharing_ablation import TASKS, sha256


def finite_json(value):
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(item) for item in value]
    return None if isinstance(value, float) and not math.isfinite(value) else value


def evaluate(config, directory):
    cfg = Config.fromfile(str(config))
    configure_target_statistics(cfg, training=False)
    # Expanded MMEngine configs sort mapping keys; retain the canonical task order.
    tasks = cfg.model.train_tasks or list(TASKS)
    epoch = cfg.train_cfg.max_epochs
    metric = cfg.default_hooks.checkpoint.save_best
    checkpoints = dict(final=directory / f"epoch_{epoch}.pth",
                       best=best_checkpoint(directory, metric))
    # Validate the fixed endpoint before writing any test result.
    final = torch.load(checkpoints["final"], map_location="cpu", weights_only=False)
    if final.get("meta", {}).get("epoch") != epoch:
        raise ValueError("Final checkpoint has not reached the requested epoch budget")
    del final
    destination = directory / "evaluation"
    destination.mkdir(exist_ok=False)
    report = dict(config=str(config.resolve()), config_sha256=sha256(config),
                  seed=cfg.randomness.seed, selected_tasks=tasks, epochs=epoch,
                  training_fingerprint=cfg.model.target_statistics["training_fingerprint"],
                  task_scales={t: cfg.model.target_statistics["tasks"][t]["std"] for t in tasks},
                  results={})
    wall = directory / "provenance/training_wall_seconds.txt"
    report["training_wall_seconds"] = int(wall.read_text()) if wall.exists() else None
    for kind, checkpoint in checkpoints.items():
        evaluation = cfg.copy()
        evaluation.work_dir = str(destination / kind)
        evaluation.load_from = str(checkpoint)
        evaluation.resume = False
        evaluation.custom_hooks = []
        evaluation.test_evaluator.evaluated_tasks = tasks
        evaluation.test_evaluator.sample_metrics_path = str(
            destination / kind / "sample_metrics.jsonl")
        start = time.monotonic()
        runner = Runner.from_cfg(evaluation)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        metrics = runner.test()
        model = runner.model.module if hasattr(runner.model, "module") else runner.model
        counts = {}
        for name, parameter in model.named_parameters():
            owner = model.parameter_owner(name) or "frozen_foundation"
            item = counts.setdefault(owner, dict(stored=0, trainable=0))
            item["stored"] += parameter.numel()
            item["trainable"] += parameter.numel() if parameter.requires_grad else 0
        report["results"][kind] = dict(
            checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint), metrics=metrics,
            sample_metrics=str(destination / kind / "sample_metrics.jsonl"),
            evaluation_wall_seconds=time.monotonic() - start,
            evaluation_peak_cuda_bytes=(torch.cuda.max_memory_allocated()
                                        if torch.cuda.is_available() else None),
            parameters_by_owner=counts, adaptation=model.encoder.adaptation_settings(),
        )
        save_record(destination / "metrics.json", finite_json(report))
        del model, runner
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return finite_json(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    evaluate(args.config.resolve(), args.directory.resolve())


if __name__ == "__main__":
    main()
