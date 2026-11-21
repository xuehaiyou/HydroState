"""Wait for published data and validated probes, then submit one measured GPU evaluation."""

import argparse
import json
import math
import os
import shlex
import subprocess
import time
from pathlib import Path

import pandas as pd

from hydrostate.data.gee.contracts import atomic_json, locked


def estimate_seconds(suite, windows):
    suite = Path(suite)
    progress = json.loads((suite / "probe_progress.json").read_text())
    if not progress["complete"]:
        raise ValueError("Model probes incomplete")
    durations = []
    for row in progress["models"]:
        bench = json.loads((suite / f"{row['name']}_benchmark.json").read_text())
        smoke = json.loads((suite / "probe" / row["name"] / "metrics.json").read_text())
        measured = next(
            r for r in bench["measurements"] if r["batch_size"] == bench["selected_batch_size"]
        )
        per_window = max(
            measured["seconds"] / measured["batch_size"],
            smoke["inference_seconds"] / smoke["windows"],
        )
        overhead = max(0, row["seconds"] - smoke["inference_seconds"])
        durations.append(
            dict(
                name=row["name"],
                per_window_seconds=per_window,
                fixed_seconds=overhead,
                estimated_seconds=overhead + windows * per_window,
            )
        )
    seconds = math.ceil(sum(r["estimated_seconds"] for r in durations) * 1.1 / 60) * 60
    return seconds, durations


def launch(root, models, output, python, probe_job, *, wait=False):
    root, models, output = Path(root), Path(models), Path(output)
    suite = output / "suite"
    with locked(output / ".launch.lock", blocking=False):
        receipt = output / "full_submission.json"
        if receipt.exists():
            return json.loads(receipt.read_text())
        while True:
            state = subprocess.run(
                ["sacct", "-X", "-n", "-j", str(probe_job), "-o", "State"],
                check=True,
                text=True,
                capture_output=True,
            ).stdout.strip()
            if any(
                s in state for s in ("FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL")
            ):
                raise RuntimeError(f"GPU probe failed: {state}")
            publication = root / "_PUBLISHED.json"
            progress = suite / "probe_progress.json"
            ready = (
                publication.exists()
                and progress.exists()
                and json.loads(progress.read_text()).get("complete")
                and "COMPLETED" in state
            )
            if ready:
                break
            atomic_json(
                output / "controller_status.json",
                dict(
                    status="waiting",
                    publication_ready=publication.exists(),
                    probe_state=state,
                    updated_at=time.time(),
                ),
            )
            if not wait:
                return dict(status="waiting")
            time.sleep(60)
        marker = json.loads(publication.read_text())
        if not marker.get("complete"):
            raise ValueError("Dataset publication incomplete")
        windows = len(pd.read_parquet(root / "samples.parquet"))
        seconds, estimates = estimate_seconds(suite, windows)
        if seconds > 24 * 3600:
            raise ValueError("Measured work exceeds one day; explicitly split the model list")
        code = Path(__file__).resolve().parents[2]
        script = output / "full.sh"
        args = [
            python,
            str(code / "tools/validation/evaluate_cover_suite.py"),
            "--models",
            str(models),
            "--root",
            str(root),
            "--output",
            str(suite),
            "--phase",
            "full",
        ]
        script.write_text(
            "#!/bin/bash\nset -euo pipefail\ncd "
            + shlex.quote(str(code))
            + "\nexport PYTHONPATH="
            + shlex.quote(str(code))
            + "\nexport OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
            "BLOSC_NTHREADS=1 PYTHONUNBUFFERED=1\nexec " + shlex.join(args) + "\n"
        )
        walltime = f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:00"
        command = [
            "sbatch",
            "--parsable",
            "--job-name=hs_cover_full",
            "--partition=l40s",
            "--qos=gpu",
            "--account=geog_geors",
            "--nodes=1",
            "--ntasks=1",
            "--gres=gpu:1",
            "--cpus-per-task=4",
            "--mem=64G",
            f"--time={walltime}",
            f"--output={output}/full_%j.out",
            f"--error={output}/full_%j.err",
            str(script),
        ]
        intent = output / "full_submission_intent.json"
        if intent.exists():
            raise RuntimeError(
                "Prior submission intent without receipt; inspect Slurm before retry"
            )
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=code, text=True).strip()
        record = dict(
            command=command,
            windows=windows,
            budget_seconds=seconds,
            estimates=estimates,
            code_commit=commit,
            pid=os.getpid(),
        )
        atomic_json(intent, record)
        result = subprocess.run(command, text=True, capture_output=True)
        record.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
        if result.returncode:
            atomic_json(output / "full_submission_rejected.json", record)
            raise RuntimeError(result.stderr)
        record["job_id"] = result.stdout.strip().split(";")[0]
        atomic_json(receipt, record)
        atomic_json(
            output / "controller_status.json", dict(status="submitted", job_id=record["job_id"])
        )
        return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "models", "output", "python", "probe-job"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--wait", action="store_true")
    print(json.dumps(launch(**vars(parser.parse_args()))), flush=True)
