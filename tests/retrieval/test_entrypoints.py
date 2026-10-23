"""Protect the current entry points against stale presets and contract drift."""

import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from mmengine.config import Config

from hydrostate.cli import configure_data_contract
from hydrostate.data.retrieval.cli import resolve_config
from hydrostate.data.retrieval.contract import PROFILE, TASKS, VERSION
from hydrostate.data.retrieval.release import verify
from hydrostate.storage.packs import file_hash
from hydrostate.tasks import resolve_tasks, task_config

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("entry", ["run_gee_sampling.sh", "run_gee_collection_io.sh"])
def test_real_launchers_preview_same_contract_without_remote_calls(tmp_path, entry):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    hostname = bin_dir / "hostname"
    hostname.write_text("#!/bin/sh\nprintf 'hpc2021-io1\\n'\n")
    hostname.chmod(0o755)
    source = ROOT / "tools/data/collect_gee_dataset.py"
    checksums = tmp_path / "sources.sha256"
    checksums.write_text(f"{hashlib.sha256(source.read_bytes()).hexdigest()}  {source}\n")
    output, candidates = tmp_path / "samples", tmp_path / "candidates"
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith("HYDROSTATE_") and k not in ("COLLECTION_CONFIG", "SAMPLES")
    }
    env.update(
        PATH=f"{bin_dir}:{env['PATH']}",
        HYDROSTATE_PYTHON=sys.executable,
        HYDROSTATE_CODE_ROOT=str(ROOT),
        HYDROSTATE_SOURCE_HASHES=str(checksums),
        HYDROSTATE_DATA_ROOT=str(output),
        HYDROSTATE_CANDIDATE_ROOT=str(candidates),
    )
    result = subprocess.run(
        ["bash", str(ROOT / "scripts" / entry), "--dry-run", "--samples", "8"],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=True,
    )
    cfg = json.loads(result.stdout)
    assert cfg["samples"] == 8 and cfg["start_year"] == 2017
    assert cfg["output_dir"] == str(output) and cfg["candidate_dir"] == str(candidates)
    assert not output.exists() and not candidates.exists()


def test_historical_collection_config_fails_with_actionable_error():
    args = SimpleNamespace(
        config=ROOT / "configs/experiments/legacy/data/collection/centered8d_16vars_hpc.py",
        output=None,
        samples=None,
        project=None,
    )
    with pytest.raises(ValueError, match="current retrieval collection config"):
        resolve_config(args)


@pytest.mark.parametrize("name", ["mtlora", "mtlora_shared", "mtlora_frozen_embed"])
def test_current_model_configs_and_dump_share_one_contract(tmp_path, name):
    cfg = Config.fromfile(str(ROOT / f"configs/hydrostate_{name}.py"))
    configure_data_contract(cfg)
    assert cfg.data_contract == VERSION and cfg.input_profile == PROFILE
    assert cfg.model.tasks == task_config(TASKS)
    assert task_config() == task_config(TASKS)
    saved = tmp_path / "expanded.py"
    cfg.dump(str(saved))
    restored = Config.fromfile(str(saved))
    configure_data_contract(restored)
    assert tuple(resolve_tasks(restored.tasks)) == tuple(resolve_tasks(cfg.tasks))


@pytest.mark.parametrize("change", ["version", "profile", "tasks", "dataset"])
def test_current_config_rejects_contract_drift(change):
    cfg = Config.fromfile(str(ROOT / "configs/hydrostate_mtlora.py"))
    if change == "version":
        cfg.data_contract = "old"
    elif change == "profile":
        cfg.input_profile = "olmoearth_l1_v1"
    elif change == "tasks":
        cfg.tasks = {"water": dict(support="pixel", grid="input")}
    else:
        cfg.train_dataloader.dataset.type = "HydroStateZarrDataset"
    with pytest.raises(ValueError):
        configure_data_contract(cfg)


def test_archived_training_presets_still_resolve():
    for path in sorted((ROOT / "configs/experiments/legacy").glob("hydrostate_*.py")):
        cfg = Config.fromfile(str(path))
        assert cfg.model.type and cfg.train_dataloader.dataset.type == "HydroStateZarrDataset"


def test_current_package_does_not_load_historical_workflows():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import hydrostate.data.retrieval.cli; import sys; "
            "assert not any(k.startswith('hydrostate.legacy') for k in sys.modules)",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_release_verification_rejects_wrong_manifest_and_contract(tmp_path):
    pd.DataFrame(dict(sample_id=["a", "b"])).to_parquet(tmp_path / "samples.parquet")
    digest = file_hash(tmp_path / "samples.parquet")
    marker = dict(
        complete=True,
        completed_at="2026-10-22",
        version=VERSION,
        input_profile=PROFILE,
        samples=2,
        manifest_sha256=digest,
    )
    stats = tmp_path / "target_stats.json"
    stats.write_text(json.dumps(dict(manifest_sha256=digest)))
    marker_path = tmp_path / "_COLLECTION_SUCCESS.json"
    marker_path.write_text(json.dumps(marker))
    assert verify(tmp_path, stats)["samples"] == 2
    for field, value in (
        ("version", "old"),
        ("complete", False),
        ("input_profile", "old"),
        ("manifest_sha256", "0" * 64),
    ):
        invalid = copy.deepcopy(marker)
        invalid[field] = value
        marker_path.write_text(json.dumps(invalid))
        with pytest.raises(ValueError):
            verify(tmp_path, stats)
    marker_path.write_text(json.dumps(marker))
    stats.write_text(json.dumps(dict(manifest_sha256="0" * 64)))
    with pytest.raises(ValueError, match="manifest mismatch"):
        verify(tmp_path, stats)
