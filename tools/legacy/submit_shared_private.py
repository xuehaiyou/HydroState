#!/usr/bin/env python
"""Submit the two shared/private experiments from an immutable HPC checkout."""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from mmengine.config import Config

from hydrostate.cli import configure_target_statistics


def save_record(path, record):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(path)


def submit_stage(record, path, stage, command, environment):
    """Persist intent before sbatch and its response before further submissions."""
    if stage in record["jobs"]:
        raise ValueError(f"Already recorded submission: {stage}")
    entry = dict(command=command, submitted_utc=datetime.now(UTC).isoformat(),
                 status="submitting")
    record["jobs"][stage] = entry
    save_record(path, record)
    result = subprocess.run(command, env=environment, text=True, capture_output=True)
    entry.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
    entry["status"] = "rejected" if result.returncode else "accepted_response"
    save_record(path, record)
    result.check_returncode()
    response = result.stdout.strip()
    if not re.fullmatch(r"[0-9]+(?:;[A-Za-z0-9_.-]+)?", response):
        raise RuntimeError(f"Check Slurm before retrying: unrecognized response {response!r}")
    job_id, _, cluster = response.partition(";")
    entry.update(job_id=job_id, cluster=cluster or None, status="submitted")
    save_record(path, record)
    print(f"{stage}: {response}", flush=True)
    return job_id


def best_checkpoint(directory, metric):
    if not re.fullmatch(r"[a-z_]+", metric):
        raise ValueError("Invalid checkpoint metric")
    paths = list(Path(directory).glob(f"best_{metric}_epoch_*.pth"))
    if len(paths) != 1:
        raise ValueError(f"Expected one best {metric} checkpoint in {directory}, got {len(paths)}")
    return paths[0].resolve()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--statistics", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--l2-pack", default="inputs/landsat_l2_sr_v1")
    args = parser.parse_args()
    code = Path(__file__).resolve().parents[2]
    root = args.root.resolve()
    if root.exists():
        raise FileExistsError(f"Existing run: inspect {root / 'submission.json'} before retrying")
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--"], cwd=code, check=True,
                   stdout=subprocess.DEVNULL)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=code, text=True).strip()
    for name in ("config.json", "weights.pth"):
        if not (args.model_path / name).is_file():
            raise FileNotFoundError(args.model_path / name)
    environment = {k: v for k, v in os.environ.items() if not k.startswith("HYDROSTATE_")}
    environment.update(
        HYDROSTATE_CODE_ROOT=str(code), HYDROSTATE_CODE_COMMIT=commit,
        HYDROSTATE_RUN_ROOT=str(root), HYDROSTATE_DATA_ROOT=str(args.data_root.resolve()),
        HYDROSTATE_MANIFEST=str(args.manifest.resolve()),
        HYDROSTATE_TARGET_STATS=str(args.statistics.resolve()),
        HYDROSTATE_MODEL_PATH=str(args.model_path.resolve()), HYDROSTATE_L2_PACK=args.l2_pack,
        HYDROSTATE_SOURCE_HASHES=str(root / "provenance/source.sha256"),
        HYDROSTATE_CONDA_BASE="/home/xiaoz/miniconda",
    )
    for key in list(os.environ):
        if key.startswith("HYDROSTATE_"):
            del os.environ[key]
    os.environ.update({k: v for k, v in environment.items() if k.startswith("HYDROSTATE_")})
    cfg = Config.fromfile(str(code / "configs/experiments/legacy/hydrostate_shared_private.py"))
    configure_target_statistics(cfg, training=True)
    if (cfg.model.type != "SharedPrivateHydroStateModel"
            or set(cfg.model.tasks) != {"water", "soil_moisture", "et"}
            or cfg.model.encoder.shared_rank != 8 or cfg.model.encoder.private_rank != 8):
        raise ValueError("Unexpected shared/private training configuration")
    root.mkdir(parents=True, exist_ok=False)
    for name in ("provenance", "logs", "bundles", "configs"):
        (root / name).mkdir()
    statistics = root / "provenance/target_statistics.json"
    shutil.copyfile(args.statistics, statistics)
    environment["HYDROSTATE_TARGET_STATS"] = str(statistics)
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=code).decode().split("\0")
    hashes = []
    for name in filter(None, tracked):
        hashes.append(f"{hashlib.sha256((code / name).read_bytes()).hexdigest()}  {name}\n")
    (root / "provenance/source.sha256").write_text("".join(hashes))
    cfg.dump(str(root / "configs/stage_a_preflight.py"))
    record = dict(
        commit=commit, code_root=str(code), run_root=str(root),
        manifest=str(args.manifest.resolve()),
        manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        statistics=str(statistics),
        statistics_sha256=hashlib.sha256(statistics.read_bytes()).hexdigest(),
        training_fingerprint=cfg.model.target_statistics["training_fingerprint"],
        environment={k: v for k, v in environment.items() if k.startswith("HYDROSTATE_")},
        gpu_resources=dict(partition="l40s", account="geog_geors", qos="gpu", gpus=1,
                           cpus=8, mem="64G", time="96:00:00"),
        experiments=dict(joint=dict(epochs=50), staged=dict(shared_epochs=30, private_epochs=20)),
        batch_size=2, accumulative_counts=8, seed=cfg.randomness.seed, jobs={},
    )
    record_path = root / "submission.json"
    save_record(record_path, record)
    for stage in ("joint50", "shared30", "water", "soil_moisture", "et", "compose"):
        command = ["sbatch", "--parsable", f"--chdir={code}",
                   f"--job-name=hs_sp_{stage}_{root.name}",
                   f"--output={root}/logs/{stage}-%j.out",
                   f"--error={root}/logs/{stage}-%j.err", "--export=ALL"]
        if stage in ("water", "soil_moisture", "et"):
            command += ["--dependency=afterok:" + record["jobs"]["shared30"]["job_id"],
                        "--kill-on-invalid-dep=yes"]
        if stage == "compose":
            dependencies = ":".join(record["jobs"][t]["job_id"]
                                    for t in ("water", "soil_moisture", "et"))
            command += [f"--dependency=afterok:{dependencies}", "--kill-on-invalid-dep=yes",
                        "scripts/legacy/slurm/34_compose_shared_private.sh"]
        else:
            command += ["scripts/legacy/slurm/33_train_shared_private.sh", stage]
        submit_stage(record, record_path, stage, command, environment)
    print(f"Submission record: {record_path}", flush=True)


if __name__ == "__main__":
    main()
