"""Prepare matched B arms and submit independent GPU probes or measured training chunks."""

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

from mmengine.config import Config

from hydrostate.data.gee.contracts import atomic_json
from hydrostate.experiments.gpu_memory import has_memory_headroom
from hydrostate.experiments.manual_cover_launch import script, slurm, submit

ARMS = ("ditask", "anysat", "d2moe")


def probe(directory):
    for batch in (5, 2, 1):
        report = directory / f"bench_{batch}.json"
        if not report.exists():
            subprocess.run([sys.executable, "tools/benchmarks/joint_training.py", "--config",
                            str(directory / "config.py"), "--output", str(report), "--batch-size",
                            str(batch), "--steps", "4", "--validation-batch", "8"], check=True)
        value = json.loads(report.read_text())
        if value["status"] == "passed" and has_memory_headroom(value):
            atomic_json(directory / "accepted.json", dict(batch=batch, report=str(report)))
            return
    raise RuntimeError("No stable microbatch among 5, 2 and 1")


def launch(args):
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    for arm in args.arms:
        directory = args.output / arm
        directory.mkdir(parents=True, exist_ok=True)
        cfg = Config.fromfile(args.reference)
        assert (cfg.model.type == "CoSAHydroStateModel" and not cfg.model.router.enabled
                and cfg.train_cfg.max_epochs == 6 and cfg.load_from is None and not cfg.resume
                and cfg.train_dataloader.dataset.type == "ManualJointDataset")
        variant = Config.fromfile(f"configs/hydrostate_joint_{arm}.py").model
        cfg.model.type = variant.type
        if arm == "anysat":
            cfg.model.encoder = variant.encoder
        else:
            cfg.model.encoder.update(type=variant.encoder.type, method=arm,
                                     gradient_checkpointing=True)
        cfg.work_dir = str(directory / "work")
        if args.phase == "prepare":
            cfg.dump(str(directory / "config.py"))
            path = directory / "probe.sh"
            script(path, [[sys.executable, "-m",
                           "hydrostate.experiments.foundation_search", "probe", str(directory)]])
            cmd = slurm(directory, "hs_probe_" + arm, 25, debug=True)
            print(arm, submit(directory, [*cmd, str(path)], dict(commit=sha)), flush=True)
            continue
        accepted = json.loads((directory / "accepted.json").read_text())
        stats = json.loads(Path(accepted["report"]).read_text())
        batch = accepted["batch"]
        cfg.train_dataloader.batch_size = batch
        cfg.optim_wrapper.accumulative_counts = 10 // batch
        cfg.param_scheduler[0].end = 100 * (10 // batch)
        # Include the cold-I/O allowance used for the manual baseline experiments.
        seconds = 2 * (stats["train_samples"] * stats["train_seconds_per_sample"]
                       + stats["validation_samples"] * stats["validation_seconds_per_sample"])
        init = stats["elapsed_seconds"]
        per_chunk = math.floor((86400 / 1.1 - init) / seconds)
        if per_chunk < 1:
            raise ValueError("Measured epoch exceeds one allocation")
        previous = None
        for start in range(0, 6, per_chunk):
            end = min(6, start + per_chunk)
            stage = directory / f"epochs_{start + 1}_{end}"
            stage.mkdir(exist_ok=True)
            cfg.train_cfg.max_epochs = end
            if start:
                cfg.resume = True
                cfg.load_from = str(Path(cfg.work_dir) / f"epoch_{start}.pth")
            cfg.dump(str(stage / "config.py"))
            path = stage / "train.sh"
            script(path, [[sys.executable, "tools/train.py", str(stage / "config.py")]])
            minutes = math.ceil(1.1 * (init + seconds * (end - start)) / 60)
            cmd = slurm(stage, "hs_" + arm, minutes, dependency=previous)
            record = dict(commit=sha, batch=batch, epochs=[start + 1, end])
            previous = submit(stage, [*cmd, str(path)], record)
            print(arm, start + 1, end, previous, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "probe", "submit"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=ARMS)
    options = parser.parse_args()
    if options.phase == "probe":
        probe(options.output)
    else:
        launch(options)
