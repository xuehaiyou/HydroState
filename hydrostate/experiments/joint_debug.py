"""Bounded real-data batch selection and paired CoSA acceptance in one debug job."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from hydrostate.data.gee.contracts import atomic_json

from .gpu_memory import has_memory_headroom

VARIANTS = dict(b="configs/hydrostate_joint_independent.py", c="configs/hydrostate_joint.py")


def execute(root, args, name, report, *, failure=None):
    print(f"Starting {name}", flush=True)
    with (root / "logs" / f"{name}.log").open("x") as log:
        result = subprocess.run([sys.executable, *args], stdout=log, stderr=subprocess.STDOUT)
    if report.is_file():
        evidence = json.loads(report.read_text())
    elif failure is not None and failure.is_file():
        evidence = json.loads(failure.read_text())
    else:
        result.check_returncode()
        raise FileNotFoundError(report)
    if evidence.get("status") != "oom":
        result.check_returncode()
    print(f"Completed {name}: {evidence.get('status')}", flush=True)
    return evidence


def safe(report):
    if report["status"] == "oom":
        return False
    if report["status"] != "passed":
        raise ValueError(f"Unexpected debug failure: {report}")
    return has_memory_headroom(report)


def run(root, *, probe_batches=(24, 28, 32, 36, 40), workers=2):
    root = Path(root).resolve()
    if (not probe_batches or tuple(sorted(set(probe_batches))) != tuple(probe_batches)
            or probe_batches[0] < 1 or workers < 0):
        raise ValueError("Positive strictly increasing candidates and nonnegative workers required")
    (root / "logs").mkdir(parents=True, exist_ok=True)
    with (root / "debug.lock").open("x") as handle:
        handle.write("One debug attempt per run directory\n")
    execute(root, ["tools/data/verify_joint_training.py", "--config", VARIANTS["c"],
                   "--output", str(root / "preflight.json"), "--check-batch"],
            "preflight", root / "preflight.json")
    candidates = []
    for batch in probe_batches:
        name = f"probe_c_{batch}"
        report = root / f"{name}.json"
        result = execute(root, ["tools/benchmarks/joint_training.py", "--config", VARIANTS["c"],
                               "--output", str(report), "--batch-size", str(batch),
                               "--steps", "4", "--workers", str(workers)], name, report)
        if not safe(result):
            break
        candidates.append(batch)
    for batch in reversed(candidates):
        trial = root / f"batch_{batch}"
        trial.mkdir()
        accepted = True
        for variant, config in VARIANTS.items():
            report = trial / f"benchmark_{variant}.json"
            result = execute(root, ["tools/benchmarks/joint_training.py", "--config", config,
                                   "--output", str(report), "--batch-size", str(batch),
                                   "--steps", "8", "--workers", str(workers)],
                             f"benchmark_{variant}_{batch}", report)
            if not safe(result):
                accepted = False
                break
        if not accepted:
            continue
        for variant, config in VARIANTS.items():
            output = trial / f"smoke_{variant}"
            result = execute(root, ["tools/experiments/verify_joint_gpu.py", "--config", config,
                                   "--output", str(output), "--batch-size", str(batch),
                                   "--train-steps", "4", "--workers", str(workers)],
                             f"smoke_{variant}_{batch}", output / "result.json",
                             failure=output / "failure.json")
            if not safe(result):
                accepted = False
                break
        if not accepted:
            continue
        for name in ("benchmark_b.json", "benchmark_c.json", "smoke_b", "smoke_c"):
            (root / name).symlink_to(Path(trial.name) / name)
        evidence = dict(status="passed", micro_batch=batch, accumulation=1,
                        effective_batch=batch, variants=list(VARIANTS), workers=workers,
                        probe_batches=list(probe_batches), selected_evidence=trial.name)
        atomic_json(root / "debug_passed.json", evidence)
        print(f"DEBUG PASSED: matched batch {batch}", flush=True)
        return evidence
    raise RuntimeError("No candidate passed both variants; inspect retained debug evidence")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--probe-batches", nargs="+", type=int, default=[24, 28, 32, 36, 40])
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    run(args.run_root, probe_batches=args.probe_batches, workers=args.workers)
