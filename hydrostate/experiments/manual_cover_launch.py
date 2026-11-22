"""Fail-closed data-to-training controller for the three approved cover candidates."""

import argparse
import copy
import json
import math
import shlex
import subprocess
import sys
import time
from pathlib import Path

from mmengine.config import Config

from hydrostate.data.gee.contracts import atomic_json, locked
from hydrostate.experiments.gpu_memory import has_memory_headroom

from .manual_cover import RESIDUAL_VARIANTS, VARIANTS, prepare, prepare_residual_pair


def submit(directory, command, record):
    receipt = directory / "submission.json"
    if receipt.exists():
        return json.loads(receipt.read_text())["job_id"]
    intent = directory / "submission_intent.json"
    if intent.exists():
        raise RuntimeError("Unresolved submission intent; inspect Slurm before retrying")
    atomic_json(intent, dict(command=command, **record))
    job = subprocess.check_output(command, text=True).strip().split(";")[0]
    atomic_json(receipt, dict(job_id=job, command=command, **record))
    return job


def script(path, commands):
    repo = Path(__file__).resolve().parents[2]
    path.write_text(
        "\n".join(
            [
                "#!/bin/bash",
                "set -euo pipefail",
                "cd " + shlex.quote(str(repo)),
                "export PYTHONPATH=" + shlex.quote(str(repo)),
                "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1",
                "export MKL_NUM_THREADS=1 BLOSC_NTHREADS=1",
                "export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
                *[shlex.join(c) for c in commands],
            ]
        )
        + "\n"
    )
    subprocess.run(["bash", "-n", str(path)], check=True)


def slurm(directory, name, minutes, debug=False, dependency=None):
    cmd = [
        "sbatch",
        "--parsable",
        "--job-name=" + name,
        "--partition=l40s",
        "--qos=" + ("debug" if debug else "gpu"),
        "--account=geog_geors",
        "--nodes=1",
        "--ntasks=1",
        "--gres=gpu:1",
        "--cpus-per-task=4",
        "--mem=64G",
        f"--time={minutes // 60:02d}:{minutes % 60:02d}:00",
        "--output=" + str(directory / "%j.out"),
        "--error=" + str(directory / "%j.err"),
    ]
    if dependency:
        cmd.append("--dependency=afterok:" + dependency)
    return cmd


def acceptance(config, output):

    directory = Path(output)
    for batch in (5, 2):
        report = directory / f"benchmark_{batch}.json"
        if not report.exists():
            # Each candidate gets a fresh subprocess after OOM; no dirty allocator/model reuse.
            command = [
                sys.executable,
                "tools/benchmarks/joint_training.py",
                "--config",
                config,
                "--output",
                str(report),
                "--batch-size",
                str(batch),
                "--steps",
                "12",
                "--workers",
                "2",
            ]
            subprocess.run(command, check=True)
        value = json.loads(report.read_text())
        if value["status"] == "passed" and has_memory_headroom(value):
            cfg = Config.fromfile(config)
            cfg.train_dataloader.batch_size = batch
            cfg.optim_wrapper.accumulative_counts = 10 // batch
            cfg.param_scheduler[0].end = 100 * (10 // batch)
            cfg.dump(str(directory / "accepted_config.py"))
            atomic_json(directory / "passed.json", dict(batch_size=batch, benchmark=str(report)))
            return
    raise RuntimeError("No tested batch passed headroom; no formal training submitted")


def finish_test(config):
    cfg = Config.fromfile(config)
    files = list(Path(cfg.work_dir).glob("best_manual_joint_score_epoch_*.pth"))
    if len(files) != 1:
        raise ValueError("Expected exactly one validation-selected checkpoint")
    subprocess.run(
        [
            sys.executable,
            "tools/test.py",
            config,
            str(files[0]),
            "--cfg-options",
            "resume=False",
            "work_dir=" + str(Path(cfg.work_dir).parent / "final_test"),
        ],
        check=True,
    )


def run(root, models=None, output=None, reference=None):
    root, output = Path(root), Path(output)
    reference = Path(reference) if reference is not None else None
    variants = RESIDUAL_VARIANTS if reference is not None else VARIANTS
    final_test = reference is None
    output.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[2]
    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    with locked(output / ".controller.lock", blocking=False):
        data_run = reference.parent if reference is not None else output.parent
        while not (root / "_PUBLISHED.json").exists():
            if (data_run / "data_failed.json").exists():
                raise RuntimeError("Data preparation failed; training stays unsubmitted")
            atomic_json(
                output / "status.json",
                dict(status="waiting_for_published_data", code_commit=commit),
            )
            time.sleep(60)
        if reference is not None:
            while not (reference / "prepared.json").is_file():
                ref_status = reference / "status.json"
                if (
                    ref_status.is_file()
                    and json.loads(ref_status.read_text()).get("status") == "failed"
                ):
                    raise RuntimeError(
                        "Reference data selection failed; shared training inputs unavailable"
                    )
                atomic_json(
                    output / "status.json",
                    dict(status="waiting_for_joint_selection", code_commit=commit),
                )
                time.sleep(60)
            prepare_residual_pair(reference, root, output)
        elif not (output / "prepared.json").exists():
            prepare(models, root, output)
        jobs = {}
        for name in variants:
            directory = output / name / "acceptance"
            directory.mkdir(exist_ok=True)
            command = [
                sys.executable,
                "tools/experiments/launch_manual_cover.py",
                "acceptance",
                "--config",
                str(output / name / "config.py"),
                "--output",
                str(directory),
            ]
            script(directory / "run.sh", [command])
            jobs[name] = submit(
                directory,
                slurm(directory, "hs_cover_probe_" + name, 20, debug=True)
                + [str(directory / "run.sh")],
                dict(code_commit=commit, phase="acceptance"),
            )
        atomic_json(output / "status.json", dict(status="acceptance_submitted", jobs=jobs))
        pending = dict(jobs)
        while pending:
            for name, job in list(pending.items()):
                state = subprocess.check_output(
                    ["sacct", "-n", "-P", "-j", job, "--format=JobIDRaw,State,ExitCode"], text=True
                )
                rows = [r.split("|")[:3] for r in state.splitlines()]
                main = next((r for r in rows if r[0] == job), None)
                if main and main[1] == "COMPLETED" and main[2] == "0:0":
                    pending.pop(name)
                elif main and main[1].split()[0] in {
                    "FAILED",
                    "TIMEOUT",
                    "CANCELLED",
                    "OUT_OF_MEMORY",
                }:
                    raise RuntimeError(f"Acceptance failed for {name}: {main}")
            if pending:
                time.sleep(60)
        training_jobs = {}
        for name in variants:
            directory = output / name
            accepted = directory / "acceptance"
            receipt = json.loads((accepted / "passed.json").read_text())
            bench = json.loads(Path(receipt["benchmark"]).read_text())
            cfg = Config.fromfile(str(accepted / "accepted_config.py"))
            # Prior six-epoch search demonstrated cold I/O costs beyond a short probe.
            # Retain measured per-sample rates with a conservative 2x transfer allowance.
            epoch_seconds = 2 * (
                bench["train_samples"] * bench["train_seconds_per_sample"]
                + bench["validation_samples"] * bench["validation_seconds_per_sample"]
            )
            initialization = bench["elapsed_seconds"]
            per_chunk = min(6, max(1, int((23 * 3600 / 1.1 - initialization) / epoch_seconds)))
            if (initialization + epoch_seconds) * 1.1 > 24 * 3600:
                raise RuntimeError(
                    "One epoch exceeds partition limit; need resumable iteration jobs"
                )
            previous, ids, completed = None, [], 0
            while completed < 6:
                end = min(6, completed + per_chunk)
                chunk = directory / f"epochs_{completed + 1}_{end}"
                chunk.mkdir(exist_ok=True)
                stage = copy.deepcopy(cfg)
                stage.train_cfg.max_epochs = end
                if completed:
                    stage.resume = True
                    stage.load_from = str(Path(cfg.work_dir) / f"epoch_{completed}.pth")
                stage.dump(str(chunk / "config.py"))
                commands = [[sys.executable, "tools/train.py", str(chunk / "config.py")]]
                if final_test and end == 6:
                    commands.append(
                        [
                            sys.executable,
                            "tools/experiments/launch_manual_cover.py",
                            "test",
                            "--config",
                            str(chunk / "config.py"),
                        ]
                    )
                script(chunk / "run.sh", commands)
                # Final held-out test also costs inference; use the actual test sample count.
                test_seconds = 0
                if final_test and end == 6:
                    from hydrostate.registry import DATASETS

                    test_seconds = (
                        2
                        * len(DATASETS.build(dict(cfg.test_dataloader.dataset)))
                        * bench["validation_seconds_per_sample"]
                    )
                minutes = math.ceil(
                    (initialization + epoch_seconds * (end - completed) + test_seconds) * 1.1 / 60
                )
                if minutes > 1440:
                    raise RuntimeError(
                        "Final test exceeds chunk walltime; split it before submission"
                    )
                previous = submit(
                    chunk,
                    slurm(chunk, "hs_manual_" + name, minutes, dependency=previous)
                    + [str(chunk / "run.sh")],
                    dict(
                        code_commit=commit,
                        batch_size=receipt["batch_size"],
                        effective_batch=10,
                        epochs=[completed + 1, end],
                        budget_minutes=minutes,
                    ),
                )
                ids.append(previous)
                completed = end
            training_jobs[name] = ids
        atomic_json(
            output / "status.json",
            dict(status="training_submitted", jobs=training_jobs, code_commit=commit),
        )
        print(json.dumps(training_jobs), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run")
    for name in ("root", "models", "output"):
        p.add_argument("--" + name, required=True)
    p = sub.add_parser("residual-pair", help="Train only the two new architectures")
    for name in ("root", "reference", "output"):
        p.add_argument("--" + name, required=True)
    p = sub.add_parser("acceptance")
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("test")
    p.add_argument("--config", required=True)
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "residual-pair":
        reference, output = (Path(args[k]).resolve() for k in ("reference", "output"))
        if reference == output or reference in output.parents or output in reference.parents:
            parser.error("Use a separate run directory for the new models")
    try:
        {"run": run, "residual-pair": run, "acceptance": acceptance, "test": finish_test}[command](
            **args
        )
    except Exception as error:
        if command in ("run", "residual-pair"):
            atomic_json(
                Path(args["output"]) / "status.json", dict(status="failed", error=str(error))
            )
        raise
