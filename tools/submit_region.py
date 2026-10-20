"""Verify prepared regional inputs and submit once from an immutable code snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from hydrostate.data.gee_sampling import atomic_json, fingerprint
from hydrostate.data.regional.common import assemble_inputs
from hydrostate.storage.packs import file_hash
from hydrostate.storage.scene_cache import SceneCache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--dates", nargs="+")
    parser.add_argument("--drop-s2", action="store_true")
    args = parser.parse_args()
    code = Path(os.environ["HYDROSTATE_CODE_ROOT"])
    commit = os.environ["HYDROSTATE_CODE_COMMIT"]
    root = args.root.resolve()
    plan = json.loads((root / "plan.json").read_text())
    dates = args.dates or [p["time"] for p in plan["periods"]]
    if not set(dates).issubset({p["time"] for p in plan["periods"]}):
        parser.error("Dates must be in plan.json")
    cache = SceneCache(root, plan)
    manifests = {}
    for day in dates:
        path = root / "manifests" / f"{day}.parquet"
        table = pd.read_parquet(path)
        accepted = table.loc[table.accepted]
        if not accepted.empty:
            for row in (accepted.iloc[0].to_dict(), accepted.iloc[-1].to_dict()):
                assemble_inputs(cache, row, drop_s2=args.drop_s2)
        manifests[day] = file_hash(path)
    for key in ("checkpoint", "target_statistics", "training_manifest"):
        if not Path(plan["config"][key]).is_file():
            raise FileNotFoundError(plan["config"][key])
    if (
        file_hash(plan["config"]["training_manifest"])
        != plan["identity"]["training_manifest_sha256"]
    ):
        raise ValueError("Training overlap metadata uses a different training manifest")
    identity = dict(
        commit=commit,
        plan_hash=plan["plan_hash"],
        manifests=manifests,
        drop_s2=args.drop_s2,
        checkpoint_sha256=file_hash(plan["config"]["checkpoint"]),
    )
    record = root / "submissions" / f"predict_{fingerprint(identity)[:20]}.json"
    record.parent.mkdir(parents=True, exist_ok=True)
    state = dict(
        identity=identity,
        created_at=datetime.now(UTC).isoformat(),
        status="prepared",
        code_root=str(code),
        root=str(root),
        dates=dates,
    )
    with record.open("x") as handle:
        json.dump(state, handle, indent=2)
    hashes = record.with_suffix(".sha256")
    with hashes.open("w") as handle:
        for path in sorted(
            p for p in code.rglob("*") if p.is_file() and "__pycache__" not in p.parts
        ):
            handle.write(
                f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(code)}\n"
            )
    env = dict(os.environ, HYDROSTATE_SOURCE_HASHES=str(hashes))
    command = [
        "sbatch",
        "--parsable",
        f"--chdir={code}",
        f"--comment=HydroState:region:{fingerprint(identity)[:20]}",
        str(code / "scripts/slurm/30_predict_region.sh"),
        "--root",
        str(root),
        "--dates",
        *dates,
    ]
    if args.drop_s2:
        command.append("--drop-s2")
    state["command"] = command
    atomic_json(record, state)
    response = subprocess.run(command, env=env, text=True, capture_output=True)
    if response.returncode:
        state.update(status="rejected", stderr=response.stderr)
        atomic_json(record, state)
        raise RuntimeError(response.stderr)
    state.update(
        status="submitted",
        slurm_response=response.stdout.strip(),
        job_id=response.stdout.strip().split(";")[0],
    )
    atomic_json(record, state)
    print(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
