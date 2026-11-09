"""Submit the two matched CoSA runs only after their real-data debug checks pass."""

import argparse
import json
import math
import os
import shlex
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

from hydrostate.data.gee.contracts import atomic_json

from .retrieval_launch import training_segments


def measurements(run, epochs):
    """Extrapolate measured I/O + computation and retain measured fixed smoke overhead."""
    passed = json.loads((run / "debug_passed.json").read_text())
    if passed["status"] != "passed" or passed["accumulation"] != 1:
        raise ValueError("Expected successful matched debug with accumulation one")
    preflight = json.loads((run / "preflight.json").read_text())
    if preflight["status"] != "passed" or not math.isclose(preflight["site_score_weight"], 2 / 3):
        raise ValueError("Missing station-priority joint preflight")
    reports = {}
    for variant in ("b", "c"):
        bench = json.loads((run / f"benchmark_{variant}.json").read_text())
        smoke = json.loads((run / f"smoke_{variant}" / "result.json").read_text())
        if (bench["status"] != "passed" or smoke["status"] != "passed"
                or bench["batch_size"] != passed["micro_batch"]
                or smoke["micro_batch"] != passed["micro_batch"]
                or smoke["model"] != "CoSAHydroStateModel"
                or smoke["site_loss_weights"] != {"site_sm": 2., "site_et": 2.}):
            raise ValueError(f"Unverified matched CoSA recipe: {variant}")
        peak = max(bench["peak_allocated_gib"], bench["peak_reserved_gib"],
                   smoke["peak_allocated_gib"], smoke["peak_reserved_gib"])
        if peak > bench["gpu_total_gib"] * .9:
            raise ValueError(f"Insufficient GPU memory headroom: {variant}")
        train_cost = bench["train_seconds_per_sample"]
        val_cost = bench["validation_seconds_per_sample"]
        if any(not math.isfinite(v) or v <= 0 for v in (train_cost, val_cost)):
            raise ValueError("Positive finite throughput measurements are required")
        selection = smoke["selection"]
        simulated = (selection["train"]["samples"] * train_cost
                     + sum(selection[s]["samples"] for s in ("val", "test")) * val_cost)
        # Includes initialization, checkpoint writing and metric bookkeeping. Using
        # it every epoch is a conservative measured overhead, not a fixed hours margin.
        overhead = max(0., smoke["elapsed_seconds"] - simulated)
        epoch = (bench["train_samples"] * train_cost
                 + bench["validation_samples"] * val_cost + overhead)
        final_test = sum(preflight["splits"]["test"].get(k, 0) for k in ("products", "sites"))
        amortized = epoch + (final_test * val_cost + overhead) / epochs
        reports[variant] = dict(epoch_seconds=epoch, fixed_overhead_seconds=overhead,
                                budget_epoch_seconds=amortized, final_test_samples=final_test,
                                train_seconds_per_sample=train_cost,
                                validation_seconds_per_sample=val_cost,
                                peak_gib=peak, gpu=bench["gpu"])
    return passed, reports


def job_script(release, config, work, end_epoch, *, resume, final, batch):
    """Training and one final test use the frozen model code and the same data contract."""
    quote = shlex.quote
    lines = ["#!/bin/bash", "set -euo pipefail", f"cd {quote(str(release))}",
             'source /home/xiaoz/miniconda/etc/profile.d/conda.sh',
             'conda activate hydrostate',
             "bash scripts/run_joint_training.sh "
             f"train_cfg.max_epochs={end_epoch} resume={resume} load_from=None"]
    if final:
        lines += ["shopt -s nullglob", f"checkpoints=({quote(str(work))}/best_joint_score_*.pth)",
                  '[[ ${#checkpoints[@]} -eq 1 ]]',
                  f"python tools/test.py {quote(config)} " + '"${checkpoints[0]}" '
                  f"--cfg-options work_dir={quote(str(work / 'final_test'))} "
                  f"test_dataloader.batch_size={batch} test_dataloader.num_workers=2"]
    return "\n".join(lines) + "\n"


def submit_after_debug(run, *, epochs=50, max_seconds=None):
    run = Path(run).resolve()
    path = run / "submission.json"
    record = json.loads(path.read_text())
    if record.get("status") == "formal_submitted":
        return record
    resources = record["resources"]
    if (not 1 <= resources["cpus_per_task"] <= 6
            or not resources["gres"].startswith("gpu:")
            or resources["gres"].rsplit(":", 1)[-1] != "1"):
        raise ValueError("Joint training requires one GPU and at most six CPU cores")
    if record["environment"]["HYDROSTATE_SLURM_PARTITION"] != resources["partition"]:
        raise ValueError("Recorded resource partition and execution environment disagree")
    limit = resources["regular_max_seconds"]
    if max_seconds is not None:
        limit = min(limit, max_seconds)
    debug = next(job for job in record["jobs"] if job["stage"] == "debug")
    requested = dict(arg[2:].split("=", 1) for arg in debug["command"]
                     if arg.startswith("--") and "=" in arg)
    for option, key in (("partition", "partition"), ("gres", "gres"),
                        ("cpus-per-task", "cpus_per_task"), ("mem", "memory")):
        if requested.get(option) != str(resources[key]):
            raise ValueError(f"Formal resources differ from the debug allocation: {option}")
    accounting = subprocess.check_output([
        "sacct", "-n", "-P", "-j", debug["job_id"], "--format=JobIDRaw,State,ExitCode",
    ], text=True)
    rows = [row.split("|") for row in accounting.splitlines()]
    job = next((row for row in rows if row[0] == debug["job_id"]), None)
    if not job or job[1:3] != ["COMPLETED", "0:0"]:
        raise ValueError(f"Debug job has not completed successfully: {job}")
    passed, reports = measurements(run, epochs)
    release = Path(record["source_release"])
    subprocess.run(["sha256sum", "--check", "--status",
                    record["environment"]["HYDROSTATE_SOURCE_HASHES"]], cwd=release, check=True)
    with (run / "formal.launch.lock").open("x") as handle:
        handle.write(datetime.now(UTC).isoformat() + "\n")
    record.update(status="formal_submitting", measurement=reports, epochs=epochs,
                  micro_batch=passed["micro_batch"], accumulation=1,
                  effective_batch=passed["micro_batch"], controller=os.environ.get(
                      "HYDROSTATE_CONTROLLER_COMMIT"),
                  timing_scope="Representative real-data extrapolation with measured overhead "
                  "and 10% margin; full-run I/O can differ")
    atomic_json(path, record)
    for variant, config in (("b", "configs/hydrostate_joint_independent.py"),
                            ("c", "configs/hydrostate_joint.py")):
        work = run / f"work_{variant}"
        segments = training_segments(reports[variant]["budget_epoch_seconds"], epochs, limit)
        previous = debug["job_id"]
        for index, segment in enumerate(segments):
            script = run / f"formal_{variant}_{index + 1}.sh"
            with script.open("x") as handle:
                handle.write(job_script(release, config, work, segment["max_epochs"],
                                        resume=index > 0, final=index == len(segments) - 1,
                                        batch=passed["micro_batch"]))
            environment = dict(record["environment"], HYDROSTATE_CONFIG=config,
                               HYDROSTATE_WORK_DIR=str(work),
                               HYDROSTATE_MICRO_BATCH=str(passed["micro_batch"]),
                               HYDROSTATE_ACCUMULATIVE_COUNTS="1")
            command = ["sbatch", "--parsable", f"--job-name=hs_cosa_joint_{variant}_{index + 1}",
                       f"--account={resources['account']}",
                       f"--partition={resources['partition']}",
                       f"--qos={resources['regular_qos']}",
                       "--nodes=1", "--ntasks=1", f"--gres={resources['gres']}",
                       f"--cpus-per-task={resources['cpus_per_task']}",
                       f"--mem={resources['memory']}", f"--time={segment['walltime']}",
                       f"--dependency=afterok:{previous}", "--kill-on-invalid-dep=yes",
                       f"--output={run}/logs/formal_{variant}_%j.out",
                       f"--error={run}/logs/formal_{variant}_%j.err", str(script)]
            intent = dict(stage="formal", variant=variant, config=config, work_dir=str(work),
                          segment=index + 1, budget=segment, command=command,
                          created_utc=datetime.now(UTC).isoformat())
            with (run / f"formal_intent_{variant}_{index + 1}.json").open("x") as handle:
                json.dump(intent, handle, indent=2)
            result = subprocess.run(command, cwd=release, env=dict(os.environ, **environment),
                                    text=True, capture_output=True)
            intent.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
            if result.returncode == 0:
                previous = intent["job_id"] = result.stdout.strip().split(";")[0]
            record["jobs"].append(intent)
            atomic_json(run / f"formal_receipt_{variant}_{index + 1}.json", intent)
            atomic_json(path, record)
            result.check_returncode()
    record["status"] = "formal_submitted"
    atomic_json(path, record)
    return record


def watch(run, *, epochs=50, interval=30):
    """CPU-light IO-node controller; writes failure instead of retrying an unknown submission."""
    run = Path(run).resolve()
    with (run / "controller.lock").open("x") as handle:
        handle.write(str(os.getpid()) + "\n")
    try:
        record = json.loads((run / "submission.json").read_text())
        debug = next(job for job in record["jobs"] if job["stage"] == "debug")["job_id"]
        while True:
            output = subprocess.check_output([
                "squeue", "-h", "-j", debug, "-o", "%T",
            ], text=True).strip()
            atomic_json(run / "controller_status.json", dict(
                pid=os.getpid(), debug_job=debug, state=output or "checking_completion",
                updated_utc=datetime.now(UTC).isoformat()))
            if not output:
                # Allow Slurm accounting to publish the terminal record.
                time.sleep(interval)
                submit_after_debug(run, epochs=epochs)
                atomic_json(run / "controller_status.json", dict(status="formal_submitted"))
                return
            time.sleep(interval)
    except Exception as error:
        atomic_json(run / "controller_failure.json", dict(error=repr(error),
                    updated_utc=datetime.now(UTC).isoformat()))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    if args.watch:
        watch(args.run_root, epochs=args.epochs)
    else:
        print(json.dumps(submit_after_debug(args.run_root, epochs=args.epochs), indent=2))
