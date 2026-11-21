"""Run an explicit, frozen model list and compare its common independent points."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from hydrostate.data.gee.contracts import atomic_json
from hydrostate.storage.packs import file_hash


def run(models, root, output, phase):
    models, root, output = Path(models), Path(root), Path(output)
    rows = json.loads(models.read_text())
    if not rows or len({r["name"] for r in rows}) != len(rows):
        raise ValueError("Model names must be unique")
    output.mkdir(parents=True, exist_ok=True)
    identity = dict(models_sha256=file_hash(models), root=str(root))
    binding = output / "suite.json"
    if binding.exists() and json.loads(binding.read_text()) != identity:
        raise ValueError("Suite model list changed")
    atomic_json(binding, identity)
    repository = Path(__file__).resolve().parents[2]
    receipts, predictions = [], []
    for row in rows:
        name = row["name"]
        if not name.replace("_", "").isalnum():
            raise ValueError("Unsafe model name")
        base = ["--root", str(root), "--config", row["config"], "--checkpoint", row["checkpoint"]]
        benchmark = output / f"{name}_benchmark.json"
        started = time.monotonic()
        if phase == "probe" and not benchmark.exists():
            subprocess.run(
                [
                    sys.executable,
                    str(repository / "tools/benchmarks/cover_points.py"),
                    *base,
                    "--output",
                    str(benchmark),
                ],
                check=True,
            )
        measured = json.loads(benchmark.read_text())
        if measured["checkpoint"] != row["checkpoint"] or measured["config"] != row["config"]:
            raise ValueError("Benchmark model identity changed")
        destination = output / phase / name
        destination.parent.mkdir(exist_ok=True)
        command = [
            sys.executable,
            str(repository / "tools/validation/evaluate_cover_points.py"),
            "predict",
            *base,
            "--output",
            str(destination),
            "--batch-size",
            str(measured["selected_batch_size"]),
        ]
        if phase == "probe":
            command += ["--limit", "6"]
        if not (destination / "_SUCCESS.json").exists():
            subprocess.run(command, check=True)
        receipt = dict(name=name, seconds=time.monotonic() - started, result=str(destination))
        receipts.append(receipt)
        atomic_json(output / f"{phase}_progress.json", dict(complete=False, models=receipts))
        predictions.append(f"{name}={destination}")
    if phase == "full":
        from .cover_points import compare

        compare(predictions, output / "common_metrics.json")
    atomic_json(output / f"{phase}_progress.json", dict(complete=True, models=receipts))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("models", "root", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--phase", choices=["probe", "full"], required=True)
    run(**vars(parser.parse_args()))
