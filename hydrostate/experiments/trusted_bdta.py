"""Run the trusted-gradient experiment on the fixed six-epoch manual-cover subset."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from mmengine.registry import init_default_scope

from hydrostate.data.gee.contracts import atomic_json
from hydrostate.datasets.trusted_sampler import TrustedSourceSampler
from hydrostate.experiments.dense_adaptation import configure as baseline_config
from hydrostate.experiments.gpu_memory import has_memory_headroom
from hydrostate.experiments.manual_cover_launch import finish_test
from hydrostate.registry import DATASETS


def configure(reference, directory):
    cfg = baseline_config(reference, directory, "bdta_unit_variance")
    if cfg.val_dataloader.batch_size != 8 or cfg.test_dataloader.batch_size != 8:
        raise ValueError("Validation and test must match the historical batch size")
    cfg.model.type = "TrustedGradientBDTA"
    cfg.model.task_loss_weights = {}
    cfg.model.site_supervision.loss_weights = {}
    cfg.custom_imports.imports.extend(
        [
            "hydrostate.models.cosa.trusted_bdta",
            "hydrostate.datasets.trusted_sampler",
            "hydrostate.engine.product_weight_counts",
        ]
    )
    cfg.custom_hooks = [*cfg.get("custom_hooks", []), dict(type="ProductWeightCountsHook")]
    cfg.train_dataloader.batch_size = 10
    cfg.train_dataloader.drop_last = False
    cfg.train_dataloader.sampler = dict(type="TrustedSourceSampler", batch_size=10)
    cfg.optim_wrapper.accumulative_counts = 1
    cfg.param_scheduler[0].end = 100
    return cfg


def run(reference, directory):
    directory.mkdir(parents=True, exist_ok=True)
    cfg = configure(reference, directory)
    path = directory / "train_config.py"
    cfg.dump(str(path))
    init_default_scope("hydrostate")
    train = DATASETS.build(dict(cfg.train_dataloader.dataset))
    sampler = TrustedSourceSampler(train, seed=cfg.randomness.seed)
    audits = [sampler.audit(sampler.batches(epoch)) for epoch in range(cfg.train_cfg.max_epochs)]
    if any(
        (a["windows"], a["batches"], a["paired_sm_batches"], a["paired_et_batches"])
        != (10912, 1092, 40, 212)
        for a in audits
    ):
        raise ValueError("Mismatch with the audited short-training subset")
    atomic_json(directory / "sampling.json", audits)
    del train, sampler
    report = directory / "benchmark.json"
    subprocess.run(
        [
            sys.executable,
            "tools/benchmarks/joint_training.py",
            "--config",
            str(path),
            "--output",
            str(report),
            "--batch-size",
            "10",
            "--validation-batch",
            "8",
            "--steps",
            "6",
            "--workers",
            "2",
        ],
        check=True,
    )
    bench = json.loads(report.read_text())
    if (
        bench["status"] != "passed"
        or not has_memory_headroom(bench)
        or bench["validation_samples"] != 3681
    ):
        raise RuntimeError(f"Real-data batch-10 preflight failed: {report}")
    subprocess.run([sys.executable, "tools/train.py", str(path)], check=True)
    finish_test(str(path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    run(args.reference, args.directory)
