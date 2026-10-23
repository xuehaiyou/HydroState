"""New labels use frozen inputs, explicit supports, times and sample-ID joins."""

import json
from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest
import torch
from mmengine.config import Config
from test_dataset import make_dataset

from hydrostate.cli import configure_data_contract, configure_target_statistics
from hydrostate.data.gee.products import BANDS, MODALITIES
from hydrostate.data.retrieval.contract import INPUT_GROUPS, RetrievalSchema
from hydrostate.data.retrieval.label_packs import (
    RetrievalLabelPackWriter,
    audit_label_packs,
    build_label_pack,
    period_bounds,
)
from hydrostate.data.retrieval.release import verify
from hydrostate.data.retrieval.specs import INPUT_SPECS, LabelSpec
from hydrostate.data.retrieval.statistics import compute_retrieval_statistics
from hydrostate.datasets.collate import hydro_collate
from hydrostate.datasets.retrieval_dataset import RetrievalDataset
from hydrostate.storage.packs import file_hash


def definitions():
    return dict(
        swe=asdict(
            LabelSpec(
                source="prepared_test_product",
                band="SWE",
                units="mm",
                support="native",
                resolution_m=500,
                qa="fixture_valid",
                aggregation="target_period_mean",
                activation="softplus",
                valid_min=0,
                capacity=2,
            )
        )
    )


def observation(manifest, sample_id="1"):
    row = pd.read_parquet(manifest).set_index("sample_id").loc[sample_id]
    weights = np.zeros((2, 40, 40), np.float32)
    weights[0, :20, :20], weights[1, 20:, 20:] = 1 / 400, 1 / 400
    return dict(
        swe=np.array([10, 20], np.float32),
        swe_valid=np.ones(2, bool),
        swe_weights=weights,
        swe_time=period_bounds(row)[0],
    )


def publish(manifest, directory="swe_v1", version="1"):
    path = manifest.parent / directory
    with RetrievalLabelPackWriter(
        path,
        manifest,
        label_specs=definitions(),
        source="synthetic fixture",
        version=version,
        shard_size=2,
    ) as writer:
        writer.write("1", observation(manifest))
        writer.finalize()
    return dict(label_specs=definitions(), label_packs=[dict(path=directory)])


def test_registry_drives_groups_bands_normalization_and_base_arrays():
    for sensor, spec in INPUT_SPECS.items():
        assert BANDS[sensor] == [band.name for band in spec.bands]
        assert MODALITIES[sensor] == spec.modality
        grouped = [
            band
            for group in INPUT_GROUPS.values()
            if group.sensor == sensor
            for band in group.bands
        ]
        assert grouped == BANDS[sensor]
    assert RetrievalSchema().targets["et_land_native_weights"].tail_shape == (128, 40, 40)


def test_join_missing_rows_flip_and_no_imagery_rewrite(tmp_path):
    manifest = make_dataset(tmp_path)
    before = {str(p): file_hash(p) for p in tmp_path.rglob("*") if p.is_file()}
    options = publish(manifest)
    assert all(file_hash(p) == digest for p, digest in before.items())
    # Reorder the manifest; the label pack still selects by sample_id.
    table = pd.read_parquet(manifest).iloc[::-1]
    table.to_parquet(manifest, index=False)
    dataset = RetrievalDataset(manifest, "train", **options)
    a, b = dataset[0], dataset[1]
    assert a["data_samples"]["swe_native_value"].tolist() == [10, 20]
    assert not b["data_samples"]["swe_native_valid"].any()
    assert b["data_samples"]["water_valid"].all()
    assert a["inputs"]["target_time"]["swe"].tolist() == observation(manifest)["swe_time"].tolist()
    batch = hydro_collate([a, b])
    assert batch["data_samples"]["swe_native_value"].shape == (2, 2)
    flipped = RetrievalDataset(manifest, "train", training=True, flip_probability=1, **options)[0]
    torch.testing.assert_close(
        flipped["data_samples"]["swe_native_weights"],
        a["data_samples"]["swe_native_weights"].flip(-1, -2),
    )
    report = audit_label_packs(
        manifest, options["label_packs"], options["label_specs"], pixel_limit=0
    )
    assert report["valid_counts"]["swe_native_valid"] == 2


@pytest.mark.parametrize("workers", [0, 2])
def test_config_statistics_and_pack_version_fingerprint(tmp_path, workers):
    manifest = make_dataset(tmp_path)
    options = publish(manifest)
    cfg = Config.fromfile("configs/hydrostate_mtlora.py")
    cfg.update(options)
    cfg.train_dataloader.dataset.update(manifest=str(manifest), data_root=str(tmp_path))
    configure_data_contract(cfg)
    assert cfg.model.heads.swe.activation == "softplus"
    assert cfg.model.tasks.swe.support == "native"
    stats = compute_retrieval_statistics(cfg, num_workers=workers)
    assert stats["tasks"]["swe"]["count"] == 2
    assert stats["tasks"]["swe"]["mean"] == 15
    cfg.model.target_statistics = stats
    configure_target_statistics(cfg, training=True)
    stats_file = tmp_path / "stats.json"
    stats_file.write_text(json.dumps(stats))
    (tmp_path / "_COLLECTION_SUCCESS.json").write_text(
        json.dumps(
            dict(
                complete=True,
                completed_at="2026-10-23",
                version=dataset_version(manifest),
                input_profile=pd.read_parquet(manifest).input_profile.iloc[0],
                samples=2,
                manifest_sha256=file_hash(manifest),
            )
        )
    )
    assert verify(tmp_path, stats_file)["samples"] == 2
    publish(manifest, "swe_v2", "2")
    cfg.label_packs = [dict(path="swe_v2")]
    for key in ("train_dataloader", "val_dataloader", "test_dataloader"):
        cfg[key].dataset.label_packs = cfg.label_packs
    with pytest.raises(ValueError, match="manifest differs"):
        configure_target_statistics(cfg, training=True)


def dataset_version(manifest):
    return pd.read_parquet(manifest).version.iloc[0]


def test_example_configuration_uses_the_same_json_as_pack_writer(tmp_path, monkeypatch):
    manifest = make_dataset(tmp_path)
    options = publish(manifest)
    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps(dict(label_specs=options["label_specs"], source="fixture", version="1"))
    )
    monkeypatch.setenv("HYDROSTATE_LABEL_SPEC", str(spec))
    monkeypatch.setenv("HYDROSTATE_LABEL_PACK", str(tmp_path / "swe_v1"))
    cfg = Config.fromfile("configs/hydrostate_extended.py")
    configure_data_contract(cfg)
    assert cfg.model.heads.swe.activation == "softplus"
    assert cfg.train_dataloader.dataset.label_specs.swe.units == "mm"
    expanded = tmp_path / "expanded.py"
    cfg.dump(str(expanded))
    restored = Config.fromfile(str(expanded))
    configure_data_contract(restored)
    assert restored.model.tasks == cfg.model.tasks


@pytest.mark.parametrize("change", ["units", "capacity", "identity", "duplicate", "missing"])
def test_reject_mismatched_specs_identity_and_sources(tmp_path, change):
    manifest = make_dataset(tmp_path)
    options = publish(manifest)
    if change == "units":
        options["label_specs"]["swe"]["units"] = "m"
    elif change == "capacity":
        options["label_specs"]["swe"]["capacity"] = 3
    elif change == "identity":
        table = pd.read_parquet(manifest)
        table.loc[1, "target_end"] = "2020-01-10"
        table.to_parquet(manifest, index=False)
    elif change == "duplicate":
        options["label_packs"] *= 2
    else:
        options["label_packs"] = []
    with pytest.raises(ValueError):
        RetrievalDataset(manifest, "train", **options)


@pytest.mark.parametrize("bad", ["time", "negative", "weights", "unknown"])
def test_failed_pack_is_not_published(tmp_path, bad):
    manifest = make_dataset(tmp_path)
    path = tmp_path / "bad"
    with pytest.raises(ValueError):
        with RetrievalLabelPackWriter(
            path, manifest, label_specs=definitions(), source="test", version="1"
        ) as writer:
            values = observation(manifest)
            if bad == "time":
                values["swe_time"][1] += 86400
            if bad == "negative":
                values["swe"][0] = -1
            if bad == "weights":
                values["swe_weights"] *= 2
            writer.write("unknown" if bad == "unknown" else "1", values)
            writer.finalize()
    assert not path.exists()
    assert not list(tmp_path.glob(".bad.*.partial"))


def test_prepared_npz_import_and_pixel_scalar_labels(tmp_path):
    manifest = make_dataset(tmp_path)
    specs = definitions()
    specs["swe"].update(support="pixel", resolution_m=30)
    specs["runoff"] = dict(specs["swe"], band="runoff", support="window_mean")
    records = []
    for sid in ("1", "0"):
        values = dict(
            swe=np.full((1, 40, 40), int(sid) + 1, np.float32),
            swe_valid=np.ones((1, 40, 40), bool),
            swe_time=observation(manifest)["swe_time"],
            runoff=np.full((1, 1, 1), int(sid) + 2, np.float32),
            runoff_valid=np.ones((1, 1, 1), bool),
            runoff_time=observation(manifest)["swe_time"],
        )
        np.savez(tmp_path / f"{sid}.npz", **values)
        records.append(dict(sample_id=sid, npz_path=f"{sid}.npz"))
    path = tmp_path / "records.parquet"
    pd.DataFrame(records).to_parquet(path, index=False)
    build_label_pack(
        manifest, path, dict(label_specs=specs, source="fixture", version="1"), tmp_path / "extra"
    )
    dataset = RetrievalDataset(
        manifest, "train", label_specs=specs, label_packs=[dict(path="extra")]
    )
    assert dataset[0]["data_samples"]["swe"].mean() == 1
    assert dataset[1]["data_samples"]["runoff"].item() == 3


@pytest.mark.parametrize("temporal", ["instant", "interval"])
def test_labels_keep_their_explicit_observation_times(tmp_path, temporal):
    manifest = make_dataset(tmp_path)
    specs = definitions()
    specs["swe"]["temporal"] = temporal
    values = observation(manifest)
    start = values["swe_time"][0] + 86400
    values["swe_time"] = np.array([start, start if temporal == "instant" else start + 86400])
    with RetrievalLabelPackWriter(
        tmp_path / "timed", manifest, label_specs=specs, source="fixture", version="1"
    ) as writer:
        writer.write("1", values)
        writer.finalize()
    dataset = RetrievalDataset(
        manifest, "train", label_specs=specs, label_packs=[dict(path="timed")]
    )
    actual = dataset[1]["inputs"]["target_time"]
    assert actual["swe"].tolist() == values["swe_time"].tolist()
    assert actual["water"].tolist() == observation(manifest)["swe_time"].tolist()
