"""Prepare matched B arms and submit independent GPU probes or six-epoch training jobs."""

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
        directory = args.output / (arm if arm == "anysat" else f"{arm}_v2")
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
            cfg.model.decoder.update(type=variant.decoder.type,
                                     num_levels=variant.decoder.num_levels,
                                     channels=variant.decoder.channels)
            cfg.model.encoder.update(type=variant.encoder.type, method=arm,
                                     out_indices=variant.encoder.out_indices,
                                     num_levels=variant.encoder.num_levels,
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
        epoch_seconds = (stats["train_samples"] * stats["train_seconds_per_sample"]
                         + stats["validation_samples"] * stats["validation_seconds_per_sample"])
        epochs = cfg.train_cfg.max_epochs
        minutes = math.ceil(1.1 * (stats["elapsed_seconds"] + epochs * epoch_seconds) / 60)
        stage = directory / "single_job"
        stage.mkdir(exist_ok=True)
        cfg.work_dir = str(stage / "work")
        path = stage / "train.sh"
        if not (stage / "submission.json").exists():
            cfg.dump(str(stage / "config.py"))
            script(path, [[sys.executable, "tools/train.py", str(stage / "config.py")]])
        cmd = slurm(stage, "hs_" + arm, minutes)
        if args.hold:
            cmd.append("--hold")
        record = dict(commit=sha, batch=batch, epochs=[1, epochs], budget_minutes=minutes)
        print(arm, submit(stage, [*cmd, str(path)], record), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "probe", "submit"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=ARMS)
    parser.add_argument("--hold", action="store_true", help="Hold replacement jobs until released")
    options = parser.parse_args()
    if options.phase == "probe":
        probe(options.output)
    else:
        launch(options)
