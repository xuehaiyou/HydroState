"""Submit a retrieval run after a successful one-epoch Slurm debug measurement."""

import argparse
import json
import math
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path


def training_segments(epoch_seconds, epochs, max_seconds=7 * 86400):
    """Use measured end-to-end epoch cost plus 10%; split at saved epoch boundaries."""
    if epoch_seconds <= 0 or epochs < 1:
        raise ValueError("Positive measured epoch duration and epoch count are required")
    capacity = int((max_seconds // 60 * 60) // (epoch_seconds * 1.1))
    if capacity < 1:
        raise ValueError("One measured epoch exceeds the regular queue limit")
    segments, completed = [], 0
    while completed < epochs:
        count = min(capacity, epochs - completed)
        minutes = math.ceil(epoch_seconds * count * 1.1 / 60)
        completed += count
        segments.append(dict(max_epochs=completed, seconds=minutes * 60,
                             walltime=f"{minutes // 60:02d}:{minutes % 60:02d}:00"))
    return segments


def inspect_debug(run, job, gpu_memory_mib):
    result = subprocess.check_output([
        "sacct", "-n", "-P", "-j", job["job_id"],
        "--format=JobIDRaw,State,ExitCode,ElapsedRaw",
    ], text=True)
    rows = [line.split("|") for line in result.splitlines()]
    row = next(r for r in rows if r[0] == job["job_id"])
    if row[1:3] != ["COMPLETED", "0:0"]:
        raise ValueError(f"Debug job did not complete successfully: {row}")
    work = Path(job["work_dir"])
    checkpoint = work / "epoch_1.pth"
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise ValueError("Debug must save a complete first epoch checkpoint")
    paths = list(work.glob("*/vis_data/scalars.json"))
    if len(paths) != 1:
        raise ValueError("Debug must have exactly one scalar log")
    scalars = [json.loads(line) for line in paths[0].read_text().splitlines() if line.strip()]
    train = [r for r in scalars if "loss" in r]
    validation = [r for r in scalars if "hydro_score" in r]
    if not train or not validation:
        raise ValueError("Debug must finish training and validation")
    for record in train:
        if any(not math.isfinite(v) for k, v in record.items()
               if k.startswith("loss") and isinstance(v, (int, float))):
            raise ValueError("Nonfinite training loss in debug")
    if not math.isfinite(validation[-1]["hydro_score"]):
        raise ValueError("Nonfinite validation score in debug")
    # MMEngine reports peak allocated MiB. Keep at least 10% of device memory free.
    memory = [r["memory"] for r in train if "memory" in r]
    if not memory or max(memory) > gpu_memory_mib * 0.9:
        raise ValueError("Debug memory measurement is missing or leaves insufficient GPU headroom")
    return dict(job_id=job["job_id"], epoch_seconds=int(row[3]),
                peak_allocated_mib=max(memory), training_records=len(train),
                validation_records=len(validation), scope="one complete train/validation epoch, "
                "including initialization, I/O and checkpoint writing")


def write_record(path, record):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(path)


def formal_command(record, debug, segment, index, previous, run):
    """Reuse the measured resources and recipe; change only duration, destination and resume."""
    command = list(debug["command"])
    script = str(Path(record["source_release"]) / "scripts/slurm/train_retrieval.sh")
    script_index = command.index(script)
    overrides = {
        "--job-name": f"hs_{run.name}_{index + 1}", "--qos": record["regular_qos"],
        "--time": segment["walltime"], "--output": str(run / "logs" / "unet_%j.out"),
        "--error": str(run / "logs" / "unet_%j.err"),
        "train_cfg.max_epochs": str(segment["max_epochs"]),
    }
    command = [f"{arg.partition('=')[0]}={overrides[arg.partition('=')[0]]}"
               if arg.partition("=")[0] in overrides else arg for arg in command]
    command[script_index:script_index] = ["--dependency=afterok:" + previous,
                                        "--kill-on-invalid-dep=yes"]
    command += ["resume=" + str(index > 0), "load_from=None"]
    return command


def submit_after_debug(run, epochs=50):
    run = Path(run).resolve()
    record_path = run / "submission.json"
    record = json.loads(record_path.read_text())
    if record.get("status") == "formal_submitted":
        return record
    debug = next(j for j in record["jobs"] if j["stage"] == "debug")
    measurement = inspect_debug(run, debug, record["gpu_memory_mib"])
    segments = training_segments(measurement["epoch_seconds"], epochs,
                                 record["regular_max_seconds"])
    # A surviving lock without a complete receipt requires checking Slurm before any retry.
    with (run / "formal.launch.lock").open("x") as handle:
        handle.write(datetime.now(UTC).isoformat() + "\n")
    release = Path(record["source_release"])
    subprocess.run(["sha256sum", "--check", "--status", record["source_hashes"]],
                   cwd=release, check=True)
    environment = dict(os.environ, **record["environment"])
    record.update(status="formal_submitting", measurement=measurement, segments=segments,
                  epochs=epochs, microbatch=debug["microbatch"], accumulation=debug["accumulation"],
                  effective_batch=debug["effective_batch"])
    write_record(record_path, record)
    previous = debug["job_id"]
    for index, segment in enumerate(segments):
        command = formal_command(record, debug, segment, index, previous, run)
        intent = dict(stage="formal", segment=index + 1, command=command,
                      created_utc=datetime.now(UTC).isoformat())
        intent_path = run / f"formal_intent_{index + 1}.json"
        with intent_path.open("x") as handle:
            json.dump(intent, handle, indent=2)
        result = subprocess.run(
            command, cwd=release, env=environment, text=True, capture_output=True)
        intent.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
        if result.returncode == 0:
            parts = result.stdout.strip().split(";")
            intent.update(job_id=parts[0], cluster=parts[1] if len(parts) > 1 else None)
            previous = intent["job_id"]
        record["jobs"].append(intent)
        write_record(run / f"formal_receipt_{index + 1}.json", intent)
        write_record(record_path, record)
        if result.returncode:
            raise RuntimeError(f"Formal submission failed; inspect {record_path}")
    record["status"] = "formal_submitted"
    write_record(record_path, record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()
    print(json.dumps(submit_after_debug(args.run_root, args.epochs), indent=2))
