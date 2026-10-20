"""Collect satellite inputs and all 16 ecosystem tasks into a new versioned root."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from mmengine.config import Config

from hydrostate.data.gee.contracts import atomic_json, locked
from hydrostate.data.gee_labels import PRODUCER_VERSION as LABEL_VERSION
from hydrostate.data.gee_labels import tasks, variables
from hydrostate.data.gee_lst import PRODUCER_VERSION as LST_VERSION
from hydrostate.data.temporal_sampling import CENTERED, sample_temporal_config

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO / "configs/data/collection/centered8d_16vars.py"


def resolve_config(args):
    cfg = Config.fromfile(args.config).to_dict()
    for key in ("samples", "project"):
        if getattr(args, key, None) is not None:
            cfg[key] = getattr(args, key)
    if args.output is not None:
        cfg["output_dir"] = str(args.output)
    for key in ("output_dir", "candidate_dir", "sampling_config"):
        path = Path(cfg[key]).expanduser()
        cfg[key] = str((path if path.is_absolute() else REPO / path).resolve())
    cfg["temporal"] = sample_temporal_config(
        json.loads(Path(cfg["sampling_config"]).read_text())
    )
    if cfg["temporal"] != dict(
        temporal_sampling=CENTERED, min_active_slots=2, min_valid=0.5, latest_days=None,
    ):
        raise ValueError(
            "This collection requires centered 4 x 8 days, >=2 slots and no recency gate"
        )
    if cfg["input_profile"] != "olmoearth_l1_v1":
        raise ValueError("This collector uses Landsat L1; L2 requires its separate input producer")
    counts = ("samples", "concurrency", "label_workers", "candidates_per_class")
    if any(type(cfg[key]) is not int or cfg[key] < 1 for key in counts):
        raise ValueError("Sample counts and worker counts must be positive integers")
    if (cfg["start_year"] > cfg["end_year"] or cfg["label_workers"] > 32
            or cfg["validation_limit"] < 0):
        raise ValueError("Invalid year range, label worker count or validation limit")
    cfg["tasks"] = list(tasks())
    cfg["label_producer"] = LABEL_VERSION
    cfg["lst_producer"] = LST_VERSION
    return cfg


def commands(cfg):
    """Use the existing producers; every writable product lives under this root."""
    root = Path(cfg["output_dir"])
    manifest = root / "samples.parquet"
    ecosystem = root / "labels/ecosystem_v1"
    lst = root / "labels/lst_observation_v1"

    def command(tool, *args):
        return [sys.executable, "-u", str(REPO / "tools/data" / tool), *map(str, args)]

    base = command(
        "prepare_gee_samples.py", "--project", cfg["project"], "--output", root,
        "--sampling-config", root / "collection_sampling.json",
        "--backend", "pixels", "--candidate-dir", cfg["candidate_dir"],
    )
    for key in ("samples", "start_year", "end_year", "seed", "split_seed",
                "candidate_seed", "candidates_per_class", "concurrency"):
        base += ["--" + key.replace("_", "-"), str(cfg[key])]
    return dict(
        samples=base,
        ecosystem=command(
            "extend_gee_labels.py", "--manifest", manifest, "--output", ecosystem,
            "--work-dir", root / "labels/.ecosystem_v1.work",
            "--project", cfg["project"], "--workers", cfg["label_workers"],
        ),
        lst=command(
            "extend_gee_lst.py", "--manifest", manifest, "--base-pack", ecosystem,
            "--output", lst, "--work-dir", root / "labels/.lst_observation_v1.work",
            "--project", cfg["project"], "--workers", cfg["label_workers"],
        ),
        validate_ecosystem=command(
            "validate_gee_labels.py", "--manifest", manifest, "--pack", ecosystem,
            "--limit", cfg["validation_limit"],
            "--report", root / "reports/ecosystem_validation.json",
        ),
        validate_lst=command(
            "validate_gee_lst.py", "--manifest", manifest, "--base-pack", ecosystem,
            "--pack", lst, "--limit", cfg["validation_limit"], "--training-limit", 64,
            "--report", root / "reports/lst_validation.json",
        ),
    )


def verify_samples(cfg):
    """Refuse partial manifests before binding immutable label packs to them."""
    import pandas as pd

    from hydrostate.data.temporal_sampling import sample_row_config
    from hydrostate.storage.packs import validate_samples

    root = Path(cfg["output_dir"])
    table = pd.read_parquet(root / "samples.parquet")
    planned = pd.read_parquet(root / "sample_index.parquet")
    validate_samples(table)
    validate_samples(planned)
    if len(table) != cfg["samples"] or set(table.sample_id) != set(planned.sample_id):
        raise ValueError("Incomplete satellite collection; label publication cannot start")
    for row in table.to_dict("records"):
        if sample_row_config(row) != cfg["temporal"]:
            raise ValueError("Sample temporal policy differs from the new collection")
        for key in ("input_shard", "label_shard"):
            shard = (root / row[key]).resolve()
            if not shard.is_relative_to(root):
                raise ValueError("A new collection must contain its own sample shards")
            marker = shard / "_SUCCESS.json"
            if not marker.is_file() or not json.loads(marker.read_text()).get("complete"):
                raise ValueError(f"Incomplete sample shard: {shard}")
    return table


def verify_pack(root, name, table):
    """A completed pack must cover this exact manifest, including missing-label rows."""
    from hydrostate.storage.packs import VariablePack, file_hash

    path = root / "labels" / name
    pack = VariablePack(path, table, kind="label")
    expected = variables() if name == "ecosystem_v1" else {"lst": None}
    version = LABEL_VERSION if name == "ecosystem_v1" else LST_VERSION
    if set(pack.metadata["variables"]) != set(expected) or pack.metadata["version"] != version:
        raise ValueError(f"Unexpected variables or producer in {path}")
    source = pack.metadata["source"]
    if (source["manifest_sha256"] != file_hash(root / "samples.parquet")
            or set(pack.locations) != set(table.sample_id)):
        raise ValueError(f"Label pack does not cover this exact completed manifest: {path}")
    if name == "lst_observation_v1" and source["base_pack_metadata_sha256"] != file_hash(
        root / "labels/ecosystem_v1/metadata.json"
    ):
        raise ValueError("LST pack was built from a different ecosystem pack")


def run_command(name, command, root):
    print(f"\n[{name}] {shlex.join(command)}", flush=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    with (root / "logs" / f"{name}.log").open("a") as log:
        log.write(f"\n{datetime.now(UTC).isoformat()} {shlex.join(command)}\n")
        log.flush()
        # Child output goes to a per-stage log; the parent remains suitable for nohup.
        try:
            subprocess.run(command, cwd=REPO, env=env, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        except subprocess.CalledProcessError:
            print(f"Stage failed; inspect {root / 'logs' / f'{name}.log'}", file=sys.stderr)
            raise


def collect(cfg):
    from hydrostate.storage.packs import file_hash

    root = Path(cfg["output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    with locked(root / ".collection.lock", blocking=False):
        frozen = root / "collection.json"
        if frozen.exists():
            if json.loads(frozen.read_text()) != cfg:
                raise ValueError("Collection configuration changed; use a new output directory")
        else:
            if any(p.name != ".collection.lock" for p in root.iterdir()):
                raise ValueError("Refusing to adopt a nonempty dataset; select a new subfolder")
            atomic_json(frozen, cfg)
        atomic_json(root / "collection_sampling.json", cfg["temporal"])
        for directory in ("logs", "reports"):
            (root / directory).mkdir(exist_ok=True)
        state_path = root / "collection_state.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        state["complete"] = False
        state.pop("completed_at", None)
        (root / "_COLLECTION_SUCCESS.json").unlink(missing_ok=True)
        atomic_json(state_path, state)
        cmds = commands(cfg)

        def run(name):
            state["active_stage"] = name
            atomic_json(state_path, state)
            run_command(name, cmds[name], root)

        manifest = root / "samples.parquet"
        if "manifest_sha256" not in state:
            run("samples")
            table = verify_samples(cfg)
            state["manifest_sha256"] = file_hash(manifest)
            atomic_json(state_path, state)
        else:
            if file_hash(manifest) != state["manifest_sha256"]:
                raise ValueError("Completed manifest changed; refusing to reuse its label packs")
            table = verify_samples(cfg)
            print("[samples] verified completed satellite collection", flush=True)

        for stage, name in (("ecosystem", "ecosystem_v1"), ("lst", "lst_observation_v1")):
            if not (root / "labels" / name).exists():
                run(stage)
            verify_pack(root, name, table)
            print(f"[{stage}] verified complete label pack", flush=True)

        # Re-run local audits after resume; downloads and published packs are reused.
        for stage in ("validate_ecosystem", "validate_lst"):
            run(stage)
        state.update(active_stage=None, complete=True, completed_at=datetime.now(UTC).isoformat())
        atomic_json(state_path, state)
        atomic_json(root / "_COLLECTION_SUCCESS.json", dict(
            samples=len(table), tasks=cfg["tasks"], input_profile=cfg["input_profile"],
            temporal=cfg["temporal"], manifest_sha256=state["manifest_sha256"],
            validation_limit=cfg["validation_limit"], completed_at=state["completed_at"],
        ))
        print(f"Collection complete: {root}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, help="New dataset subfolder")
    parser.add_argument("--samples", type=int, help="Override the config's sample count")
    parser.add_argument("--project", help="Override the Earth Engine project")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print commands; no files or GEE calls",
    )
    args = parser.parse_args()
    cfg = resolve_config(args)
    if args.dry_run:
        print(json.dumps(cfg, indent=2))
        for stage, command in commands(cfg).items():
            print(f"\n[{stage}] {shlex.join(command)}")
        return
    collect(cfg)


if __name__ == "__main__":
    main()
