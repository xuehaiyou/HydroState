"""Targets may grow without changing shard contracts or frozen training releases."""

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from hydrostate.data.gee.contracts import atomic_json, fingerprint
from hydrostate.data.retrieval.cli import DEFAULT_CONFIG, resolve_config
from hydrostate.data.retrieval.contract import PROFILE, VERSION
from hydrostate.data.retrieval.plan import (
    DATA_FUNCTIONS,
    DATA_MODULES,
    bind_plan,
    data_signature,
    legacy_producer_hash,
    target_samples,
)
from hydrostate.storage.packs import file_hash


def config(root, samples=8):
    return resolve_config(
        SimpleNamespace(config=DEFAULT_CONFIG, output=root, samples=samples, project=None)
    )


def test_extension_preserves_contract_and_archives_the_previous_release(tmp_path):
    cfg = config(tmp_path)
    original = bind_plan(tmp_path, cfg)
    identity = (tmp_path / "collection.json").read_bytes()
    manifest = tmp_path / "samples.parquet"
    manifest.write_bytes(b"unchanged manifest payload")
    digest = file_hash(manifest)
    atomic_json(
        tmp_path / "_COLLECTION_SUCCESS.json",
        dict(samples=8, manifest_sha256=digest, complete=True),
    )
    (tmp_path / "target_stats.json").write_text('{"frozen":true}')
    grown = dict(cfg, samples=16, concurrency=2, candidate_pool_capacity=64)
    assert bind_plan(tmp_path, grown) == original
    assert fingerprint(original) == fingerprint(cfg)
    assert (tmp_path / "collection.json").read_bytes() == identity
    assert target_samples(tmp_path, cfg) == 16
    assert not (tmp_path / "_COLLECTION_SUCCESS.json").exists()
    release = next((tmp_path / "releases").iterdir())
    assert (release / "samples.parquet").read_bytes() == manifest.read_bytes()
    assert json.loads((release / "collection_plan.json").read_text())["samples"] == 8
    assert (release / "target_stats.json").read_bytes() == (
        tmp_path / "target_stats.json"
    ).read_bytes()
    with pytest.raises(ValueError, match="only increase"):
        bind_plan(tmp_path, cfg)
    with pytest.raises(ValueError, match="contract changed"):
        bind_plan(tmp_path, dict(grown, ice_confidence=0.7))


def test_scheduler_adoption_checks_previous_code_and_data_dependencies(tmp_path):
    root, previous = tmp_path / "data", tmp_path / "previous"
    root.mkdir()
    repo = Path(__file__).resolve().parents[2]
    # Recreate a frozen previous scheduler with identical data-producing code.
    paths = set(DATA_MODULES) | set(DATA_FUNCTIONS)
    paths.update(
        str(p.relative_to(repo)) for p in (repo / "hydrostate/data/retrieval").glob("*.py")
    )
    for name in paths:
        destination = previous / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo / name, destination)
    scheduler = previous / "hydrostate/data/retrieval/scheduling.py"
    scheduler.write_text(scheduler.read_text() + "\n# Previous scheduler revision\n")
    cfg = config(root)
    old = dict(cfg, producer_hash=legacy_producer_hash(previous))
    atomic_json(root / "collection.json", dict(old, version=VERSION, input_profile=PROFILE))
    with pytest.raises(ValueError, match="adopt-from"):
        bind_plan(root, cfg)
    assert bind_plan(root, cfg, adopt_from=previous) == old
    assert data_signature(previous) == data_signature(repo)
    # A provenance mismatch must not be silently accepted on future resumes.
    provenance = root / "collection_producer.json"
    value = json.loads(provenance.read_text())
    value["data_producer_hash"] = "changed"
    atomic_json(provenance, value)
    with pytest.raises(ValueError, match="Data producer"):
        bind_plan(root, cfg)
