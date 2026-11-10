"""Small real-data GPU test of the unchanged joint model and MMEngine training loops."""

import argparse
import json
import math
import time
from pathlib import Path

import torch
from mmengine.config import Config
from mmengine.hooks import Hook
from mmengine.registry import init_default_scope

from hydrostate.cli import configure_pretrained_encoder, configure_target_statistics
from hydrostate.data.gee.contracts import atomic_json
from hydrostate.engine.runner import HydroStateRunner as Runner
from hydrostate.experiments.gpu_memory import CudaMemoryMeter
from hydrostate.registry import DATASETS


def representative_indices(dataset):
    """Pair product groups with site tasks/supports from each available source cohort."""
    products = [int(group.index[0]) for _, group in dataset.product.table.groupby("sampling_group")]
    if "valid_label_counts" in dataset.product.table:
        counts = dataset.product.table.valid_label_counts.map(
            lambda value: json.loads(value) if isinstance(value, str) else value)
        for task in dataset.product.tasks:
            values = counts.map(lambda row, name=task: row.get(name, 0))
            if values.max() > 0:
                products.append(int(values.idxmax()))
        products = list(dict.fromkeys(products))
    sites = []
    for component, site in enumerate(dataset.sites, start=1):
        for mask in (
            site.table.valid_sm.gt(0),
            site.table.et_support.eq("static_210m"),
            site.table.et_support.eq("footprint"),
        ):
            if mask.any():
                sites.append(dataset.offsets[component] + int(mask[mask].index[0]))
    sites = list(dict.fromkeys(sites))
    if not products or not sites:
        raise ValueError("Smoke test requires product and site examples")
    indices = []
    for i in range(max(len(products), len(sites))):
        indices.extend((products[i % len(products)], sites[i % len(sites)]))
    return indices


class FiniteLossHook(Hook):
    def __init__(self):
        self.steps = 0
        self.memory = CudaMemoryMeter()

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        if not outputs or "loss" not in outputs:
            raise ValueError("Training step did not report loss")
        if any(not math.isfinite(float(value)) for value in outputs.values()):
            raise ValueError("Nonfinite training loss or diagnostic")
        self.steps += 1
        # Read before LoggerHook resets its memory counters.
        self.memory.sample()

    def after_val_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        self.memory.sample()

    def after_test_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        self.memory.sample()


def verify_checkpoint_parameters(runner, checkpoint):
    """Perturb the disposable model so a no-op or incomplete reload cannot pass."""
    expected = {name: p.detach().cpu().clone() for name, p in runner.model.named_parameters()
                if p.requires_grad}
    if not expected or not any(value.count_nonzero() for value in expected.values()):
        raise ValueError("No nonzero trainable checkpoint parameters")
    if any(not torch.isfinite(value).all() for value in expected.values()):
        raise ValueError("Nonfinite trained checkpoint parameters")
    with torch.no_grad():
        for name, parameter in runner.model.named_parameters():
            if name in expected:
                parameter.zero_()
    runner.load_checkpoint(str(checkpoint), strict=True)
    for name, parameter in runner.model.named_parameters():
        if name in expected and not torch.equal(expected[name], parameter.detach().cpu()):
            raise ValueError(f"Checkpoint did not restore trained parameter: {name}")
    return len(expected)


def run(config, output, *, batch_size=2, workers=0, train_steps=8):
    output = Path(output)
    if min(batch_size, train_steps) < 1 or workers < 0:
        raise ValueError("Positive batch/steps and nonnegative worker count required")
    if (output / "result.json").exists():
        raise FileExistsError("Use a new smoke output directory")
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Smoke test requires exactly one CUDA GPU")
    cfg = Config.fromfile(config)
    init_default_scope("hydrostate")
    configure_pretrained_encoder(cfg)
    configure_target_statistics(cfg, training=True)
    selection = {}
    for split in ("train", "val", "test"):
        loader = cfg[f"{split}_dataloader"]
        dataset = DATASETS.build(dict(loader.dataset))
        indices = representative_indices(dataset)
        if split == "train":
            count = max(len(indices), train_steps * batch_size)
            indices = [indices[i % len(indices)] for i in range(count)]
        selection[split] = dict(indices=indices, samples=len(indices))
        loader.update(sampler=indices, batch_size=batch_size, num_workers=workers,
                      persistent_workers=False)
    cfg.work_dir = str(output)
    cfg.train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=1, val_interval=1)
    cfg.optim_wrapper.accumulative_counts = 1
    cfg.param_scheduler = []
    cfg.default_hooks.logger.interval = 1
    cfg.default_hooks.checkpoint.update(save_optimizer=False, save_param_scheduler=False)
    cfg.env_cfg.cudnn_benchmark = False
    cfg.launcher = "none"
    cfg.dump(str(output / "smoke_config.py"))
    atomic_json(output / "selection.json", selection)
    runner = Runner.from_cfg(cfg)
    hook = FiniteLossHook()
    runner.register_hook(hook, priority="ABOVE_NORMAL")
    tracked = {
        name: p.detach().cpu().clone()
        for name, p in runner.model.named_parameters()
        if p.requires_grad and "heads." in name
    }
    if not tracked:
        raise ValueError("No trainable task heads found")
    gradient_checks = []

    def check_gradient(gradient):
        if not torch.isfinite(gradient).all():
            raise ValueError("Nonfinite task-head gradient")
        gradient_checks.append(True)
        return gradient

    handles = [
        p.register_hook(check_gradient)
        for n, p in runner.model.named_parameters() if n in tracked
    ]
    torch.cuda.reset_peak_memory_stats()
    runner.train()
    changed = [
        name for name, p in runner.model.named_parameters()
        if name in tracked and not torch.equal(tracked[name], p.detach().cpu())
    ]
    if not changed or not gradient_checks or not hook.steps:
        raise ValueError("No finite-gradient optimizer update was demonstrated")
    for handle in handles:
        handle.remove()
    checkpoint = output / "epoch_1.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    restored = verify_checkpoint_parameters(runner, checkpoint)
    metrics = runner.test()
    if not math.isfinite(metrics["joint_score"]):
        raise ValueError("Nonfinite joint evaluation score")
    for task in ("site_sm", "site_et"):
        if metrics.get(f"site/{task}_count", 0) <= 0:
            raise ValueError(f"No valid {task} evaluation labels")
    result = dict(
        status="passed", scope="GPU joint training/validation/checkpoint reload/test smoke only",
        gpu=torch.cuda.get_device_name(), model=cfg.model.type,
        micro_batch=batch_size, accumulation=1, train_steps=hook.steps,
        changed_task_head_parameters=len(changed), finite_gradient_checks=len(gradient_checks),
        **hook.memory.sample(), checkpoint_parameters_verified=restored,
        elapsed_seconds=time.monotonic() - started,
        test_joint_score=metrics["joint_score"],
        site_loss_weights=dict(cfg.model.site_supervision.loss_weights),
        selection=selection, checkpoint=str(checkpoint),
        interpretation="Representative smoke subset; not accuracy or maximum-batch validation",
    )
    atomic_json(output / "result.json", result)
    print(result, flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hydrostate_joint.py")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--train-steps", type=int, default=8)
    args = parser.parse_args()
    try:
        run(args.config, args.output, batch_size=args.batch_size, workers=args.workers,
            train_steps=args.train_steps)
    except Exception as error:
        atomic_json(Path(args.output) / "failure.json", dict(
            status="oom" if isinstance(error, torch.cuda.OutOfMemoryError) else "failed",
            error=repr(error)))
        raise
