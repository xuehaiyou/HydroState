"""Isolation and restart checks for the new complete satellite/label collection."""

import json
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from hydrostate.data.gee_labels import PRODUCER_VERSION as LABEL_VERSION
from hydrostate.data.gee_labels import variables as ecosystem_variables
from hydrostate.data.gee_lst import PRODUCER_VERSION as LST_VERSION
from hydrostate.data.gee_lst import variables as lst_variables
from hydrostate.storage.packs import VariablePackWriter, file_hash
from tools.data import collect_gee_dataset as collection


def config(root, samples=2):
    return collection.resolve_config(SimpleNamespace(
        config=collection.DEFAULT_CONFIG, output=root, samples=samples, project=None,
    ))


def base_samples(cfg, count=2):
    root = collection.Path(cfg["output_dir"])
    rows = []
    for i in range(count):
        row = dict(sample_id=f"sample-{i}", split="train",
                   temporal_sampling=cfg["temporal"]["temporal_sampling"],
                   input_temporal_config=json.dumps(cfg["temporal"]),
                   input_shard=f"zarr/{i}/inputs.zarr", label_shard=f"zarr/{i}/labels.zarr")
        for key in ("input_shard", "label_shard"):
            shard = root / row[key]
            shard.mkdir(parents=True)
            (shard / "_SUCCESS.json").write_text('{"complete": true}')
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_parquet(root / "samples.parquet", index=False)
    table.to_parquet(root / "sample_index.parquet", index=False)
    return table


def pack(root, name, *, limit=None, manifest_hash=None):
    manifest = root / "samples.parquet"
    specs = ecosystem_variables() if name == "ecosystem_v1" else lst_variables()
    producer = LABEL_VERSION if name == "ecosystem_v1" else LST_VERSION
    source = dict(producer=producer, manifest_sha256=manifest_hash or file_hash(manifest))
    if name == "lst_observation_v1":
        source["base_pack_metadata_sha256"] = file_hash(root / "labels/ecosystem_v1/metadata.json")
    rows = pd.read_parquet(manifest)
    with VariablePackWriter(
        root / "labels" / name, manifest, kind="label", variables=specs,
        source=source, version=producer,
    ) as writer:
        for sid in rows.sample_id[:limit]:
            values = {}
            for key, spec in specs.items():
                values[key] = np.zeros(spec["shape"], dtype="float32")
                values[key + "_valid"] = np.zeros(spec["shape"], dtype=bool)
                if spec["grid"] == "native":
                    values[key + "_weights"] = np.zeros(
                        (*spec["shape"], *spec["spatial_shape"]), dtype="float32",
                    )
                if "uncertainty_units" in spec:
                    values[key + "_uncertainty"] = np.zeros(spec["shape"], dtype="float32")
                if "time_support" in spec:
                    values[key + "_time"] = np.zeros(2, dtype="int64")
            writer.write(sid, values)
        writer.finalize()


def test_dry_run_does_not_create_or_collect(tmp_path):
    destination = tmp_path / "new-data"
    result = subprocess.run(
        [sys.executable, str(collection.REPO / "tools/data/collect_gee_dataset.py"),
         "--output", str(destination), "--samples", "64", "--dry-run"],
        check=True, capture_output=True, text=True,
    )
    assert not destination.exists()
    assert '"min_active_slots": 2' in result.stdout
    assert '"samples": 64' in result.stdout
    assert "extend_gee_lst.py" in result.stdout
    assert "merged_v1" not in result.stdout


def test_refuses_existing_dataset_before_remote_calls(tmp_path, monkeypatch):
    (tmp_path / "samples.parquet").write_text("old data")
    monkeypatch.setattr(collection, "run_command", lambda *a: pytest.fail("remote call"))
    with pytest.raises(ValueError, match="nonempty dataset"):
        collection.collect(config(tmp_path))
    assert (tmp_path / "samples.parquet").read_text() == "old data"


@pytest.mark.parametrize("incomplete", ["count", "marker", "policy"])
def test_incomplete_base_cannot_start_label_collection(tmp_path, incomplete):
    cfg = config(tmp_path)
    table = base_samples(cfg, count=1 if incomplete == "count" else 2)
    if incomplete == "marker":
        (tmp_path / table.iloc[0].input_shard / "_SUCCESS.json").unlink()
    if incomplete == "policy":
        table["input_temporal_config"] = json.dumps(dict(cfg["temporal"], min_active_slots=4))
        table.to_parquet(tmp_path / "samples.parquet", index=False)
    with pytest.raises(ValueError):
        collection.verify_samples(cfg)


@pytest.mark.parametrize("mismatch", ["partial_pack", "manifest_hash"])
def test_rejects_wrong_published_pack(tmp_path, mismatch):
    table = base_samples(config(tmp_path))
    pack(tmp_path, "ecosystem_v1", limit=1 if mismatch == "partial_pack" else None,
         manifest_hash="wrong" if mismatch == "manifest_hash" else None)
    with pytest.raises(ValueError, match="exact completed manifest"):
        collection.verify_pack(tmp_path, "ecosystem_v1", table)


def test_interrupted_collection_resumes_and_accepts_missing_labels(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    called = []

    def execute(name, command, root):
        called.append(name)
        if name == "samples":
            base_samples(cfg)
        elif name == "ecosystem":
            pack(root, "ecosystem_v1")
        elif name == "lst":
            raise RuntimeError("transient connection error")

    monkeypatch.setattr(collection, "run_command", execute)
    with pytest.raises(RuntimeError, match="connection error"):
        collection.collect(cfg)
    assert not (tmp_path / "_COLLECTION_SUCCESS.json").exists()
    manifest_hash = file_hash(tmp_path / "samples.parquet")
    base_hash = file_hash(tmp_path / "labels/ecosystem_v1/metadata.json")

    def resume(name, command, root):
        called.append(name)
        if name == "lst":
            pack(root, "lst_observation_v1")
        elif name in ("samples", "ecosystem"):
            pytest.fail("Completed downloads must not be relaunched")

    called.clear()
    monkeypatch.setattr(collection, "run_command", resume)
    collection.collect(dict(cfg, sampling_config="/new/frozen/release/centered8d.json"))
    assert called == ["lst", "validate_ecosystem", "validate_lst"]
    assert file_hash(tmp_path / "samples.parquet") == manifest_hash
    assert file_hash(tmp_path / "labels/ecosystem_v1/metadata.json") == base_hash
    success = json.loads((tmp_path / "_COLLECTION_SUCCESS.json").read_text())
    assert success["samples"] == 2
    assert len(success["tasks"]) == 16
    assert success["temporal"]["min_active_slots"] == 2
    with pytest.raises(ValueError, match="configuration changed"):
        collection.collect(dict(cfg, samples=3))

    table = pd.read_parquet(tmp_path / "samples.parquet")
    table.loc[0, "split"] = "test"
    table.to_parquet(tmp_path / "samples.parquet", index=False)
    with pytest.raises(ValueError, match="manifest changed"):
        collection.collect(cfg)
    assert not (tmp_path / "_COLLECTION_SUCCESS.json").exists()
    assert not json.loads((tmp_path / "collection_state.json").read_text())["complete"]
