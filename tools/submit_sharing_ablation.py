#!/usr/bin/env python
"""Submit the fixed-coefficient, three-seed OlmoEarth sharing comparison."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

from mmengine.config import Config

from hydrostate.cli import configure_data_contract, configure_target_statistics
from hydrostate.data.sampling_manifest import geographic_block, read_manifest
from tools.submit_shared_private import save_record, submit_stage

TASKS = ("water", "soil_moisture", "et")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def experiment_plan(seeds):
    if not seeds or len(set(seeds)) != len(seeds) or any(s < 0 for s in seeds):
        raise ValueError("Seeds must be nonnegative and unique")
    return [dict(stage=f"{group}{'_' + task if task else ''}_s{seed}",
                 group=group, task=task, seed=seed,
                 config=f"configs/hydrostate_sharing_ablation_{group}.py")
            for seed in seeds for group, task in
            [("shared", None), ("shared_private", None),
             *(("independent", task) for task in TASKS)]]


def validate_config(cfg, entry):
    shared, private = dict(shared=(16, 0), shared_private=(8, 8), independent=(0, 16))[
        entry["group"]]
    single = entry["group"] == "independent"
    if (cfg.model.type != "SharedPrivateHydroStateModel"
            or set(cfg.model.tasks) != set(TASKS)
            or cfg.model.encoder.shared_rank != shared
            or cfg.model.encoder.private_rank != private
            or cfg.model.encoder.shared_gate_mode != "fixed_one"
            or cfg.model.encoder.lora_scale != 1.0
            or cfg.model.branch_init_seed != entry["seed"]
            or cfg.randomness.seed != entry["seed"]
            or cfg.model.train_shared != (not single)
            or cfg.model.active_tasks_only != single
            or cfg.model.train_tasks != ([entry["task"]] if single else None)
            or cfg.model.water_soil_coupling is not None
            or cfg.load_from is not None or cfg.resume
            or cfg.train_cfg.max_epochs != 50
            or cfg.train_dataloader.batch_size != 2
            or cfg.optim_wrapper.accumulative_counts != 8):
        raise ValueError(f"Unexpected comparison configuration: {entry['stage']}")
    for name in ("val_evaluator", "test_evaluator"):
        if cfg[name].get("evaluated_tasks") != ([entry["task"]] if single else None):
            raise ValueError("Independent runs must evaluate only their trained task")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "data-root", "manifest", "statistics", "model-path"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--l2-pack", default="inputs/landsat_l2_sr_v1")
    parser.add_argument("--seeds", type=int, nargs="+", default=[3407, 3408, 3409])
    args = parser.parse_args()
    plan = experiment_plan(args.seeds)
    code, root = Path(__file__).resolve().parents[1], args.root.resolve()
    if root.exists():
        raise FileExistsError(f"Existing run: inspect {root / 'submission.json'} before retrying")
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--"], cwd=code, check=True,
                   stdout=subprocess.DEVNULL)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=code, text=True).strip()
    # This manifest's split seed belongs to data construction, not model initialization.
    table = read_manifest(args.manifest, split_seed=42)
    if table.sample_id.duplicated().any() or set(table.split) != {"train", "validation", "test"}:
        raise ValueError("The three-way comparison needs unique samples and all three splits")
    audit = dict(split_seed=42, counts=table.split.value_counts().to_dict(),
                 geographic_blocks=len({geographic_block(r) for r in table.to_dict("records")}),
                 same_block_cross_split=False,
                 scope="5-degree blocks; native-footprint overlap not exhaustively audited")
    assets = [args.manifest.resolve(), *(args.model_path.resolve() / n
                                       for n in ("config.json", "weights.pth"))]
    pack = args.data_root.resolve() / args.l2_pack
    assets.extend(pack / n for n in ("_SUCCESS.json", "metadata.json", "index.parquet"))
    for path in [*assets, args.statistics]:
        if not path.is_file():
            raise FileNotFoundError(path)
    environment = {k: v for k, v in os.environ.items() if not k.startswith("HYDROSTATE_")}
    environment.update(
        HYDROSTATE_CODE_ROOT=str(code), HYDROSTATE_CODE_COMMIT=commit,
        HYDROSTATE_RUN_ROOT=str(root), HYDROSTATE_DATA_ROOT=str(args.data_root.resolve()),
        HYDROSTATE_MANIFEST=str(args.manifest.resolve()),
        HYDROSTATE_TARGET_STATS=str(args.statistics.resolve()),
        HYDROSTATE_MODEL_PATH=str(args.model_path.resolve()), HYDROSTATE_L2_PACK=args.l2_pack,
        HYDROSTATE_SOURCE_HASHES=str(root / "provenance/source.sha256"),
        HYDROSTATE_ASSET_HASHES=str(root / "provenance/assets.sha256"),
        HYDROSTATE_CONDA_BASE="/home/xiaoz/miniconda",
    )
    for key in list(os.environ):
        if key.startswith("HYDROSTATE_"):
            del os.environ[key]
    os.environ.update({k: v for k, v in environment.items() if k.startswith("HYDROSTATE_")})
    base = Config.fromfile(str(code / plan[0]["config"]))
    configure_target_statistics(base, training=True)
    root.mkdir(parents=True, exist_ok=False)
    for name in ("provenance", "logs", "configs"):
        (root / name).mkdir()
    statistics = root / "provenance/target_statistics.json"
    shutil.copyfile(args.statistics, statistics)
    environment["HYDROSTATE_TARGET_STATS"] = str(statistics)
    os.environ["HYDROSTATE_TARGET_STATS"] = str(statistics)
    assets.append(statistics)
    for entry in plan:
        os.environ["HYDROSTATE_ABLATION_SEED"] = str(entry["seed"])
        os.environ["HYDROSTATE_ABLATION_TASK"] = entry["task"] or "soil_moisture"
        os.environ["HYDROSTATE_WORK_DIR"] = str(root / entry["stage"])
        cfg = Config.fromfile(str(code / entry["config"]))
        configure_data_contract(cfg)
        if any(cfg[name].dataset != base[name].dataset
               for name in ("train_dataloader", "val_dataloader", "test_dataloader")):
            raise ValueError("All experiments must use the same data contract")
        configure_target_statistics(cfg, training=False)
        validate_config(cfg, entry)
        destination = root / "configs" / f"{entry['stage']}.py"
        cfg.dump(str(destination))
        entry.update(expanded_config=str(destination), config_sha256=sha256(destination),
                     work_dir=str(root / entry["stage"]),
                     selected_tasks=[entry["task"]] if entry["task"] else list(TASKS),
                     checkpoint_metric=f"{entry['task']}_rmse" if entry["task"] else "hydro_score")
        assets.append(destination)
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=code).decode().split("\0")
    (root / "provenance/source.sha256").write_text("".join(
        f"{sha256(code / name)}  {name}\n" for name in filter(None, tracked)))
    (root / "provenance/assets.sha256").write_text("".join(
        f"{sha256(path)}  {path}\n" for path in assets))
    (root / "provenance/split_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    record = dict(commit=commit, code_root=str(code), run_root=str(root),
                  manifest=str(args.manifest.resolve()), manifest_sha256=sha256(args.manifest),
                  statistics=str(statistics), statistics_sha256=sha256(statistics),
                  training_fingerprint=base.model.target_statistics["training_fingerprint"],
                  split_audit=audit, seeds=args.seeds, epochs=50, batch_size=2,
                  accumulative_counts=8, primary_checkpoint="epoch_50.pth",
                  secondary_checkpoint="validation-selected; independent per-task selection",
                  sharing_scope="FM LoRA updates; all readouts remain private in every group",
                  gpu_resources=dict(partition="l40s", account="geog_geors", qos="gpu", gpus=1,
                                     cpus=8, mem="64G", time="96:00:00"),
                  environment={k: v for k, v in environment.items() if k.startswith("HYDROSTATE_")},
                  experiments=plan, jobs={})
    record_path = root / "submission.json"
    save_record(record_path, record)
    for entry in plan:
        stage = entry["stage"]
        command = ["sbatch", "--parsable", f"--chdir={code}",
                   f"--job-name=hs_abl_{stage}_{root.name}",
                   f"--output={root}/logs/{stage}-%j.out",
                   f"--error={root}/logs/{stage}-%j.err", "--export=ALL",
                   "scripts/slurm/35_train_sharing_ablation.sh", stage]
        submit_stage(record, record_path, stage, command, environment)
    dependencies = ":".join(record["jobs"][entry["stage"]]["job_id"] for entry in plan)
    submit_stage(record, record_path, "summarize", [
        "sbatch", "--parsable", f"--chdir={code}", f"--job-name=hs_abl_summary_{root.name}",
        f"--output={root}/logs/summarize-%j.out", f"--error={root}/logs/summarize-%j.err",
        "--export=ALL", f"--dependency=afterok:{dependencies}", "--kill-on-invalid-dep=yes",
        "scripts/slurm/36_summarize_sharing_ablation.sh"], environment)
    print(f"Submission record: {record_path}", flush=True)


if __name__ == "__main__":
    main()
