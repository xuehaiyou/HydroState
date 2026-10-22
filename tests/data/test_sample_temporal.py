"""New sample selection, immutable legacy data, and actual timestamps end to end."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import rasterio
import torch
import zarr
from affine import Affine

from hydrostate.legacy import sampling as samples
from hydrostate.data.merge_samples import merge
from hydrostate.data.regional.common import assemble_inputs
from hydrostate.data.sampling_manifest import geographic_split
from hydrostate.data.temporal_sampling import CENTERED, LEGACY_SAMPLE_CONFIG, observation_slots
from hydrostate.models.temporal_metadata import prepare_temporal_metadata

CENTERED_CONFIG = dict(
    temporal_sampling=CENTERED, min_active_slots=2, min_valid=0.5, latest_days=None,
)
TARGET = "2025-01-01"
T = datetime(2025, 1, 1, tzinfo=UTC)


def scene(offset, key="s2", orbit=12, fraction=0.75):
    return dict(
        asset_id=f"{key}/{offset}/{orbit}",
        **{"system:time_start": int((T + timedelta(days=offset)).timestamp() * 1000)},
        relativeOrbitNumber_start=orbit, valid_fraction=fraction,
    )


def test_new_default_and_explicit_legacy_and_invalid_recency():
    args = SimpleNamespace(sampling_config=None, reingest_from=None)
    samples.resolve_sampling_config(args)
    assert {k: getattr(args, k) for k in CENTERED_CONFIG} == CENTERED_CONFIG
    assert observation_slots(TARGET, vars(args)) == [
        ("2024-12-16", "2024-12-24"), ("2024-12-24", "2025-01-01"),
        ("2025-01-01", "2025-01-09"), ("2025-01-09", "2025-01-17"),
    ]
    old = SimpleNamespace(
        sampling_config=Path("configs/experiments/legacy/data/sampling/past16d.json"), reingest_from=None,
    )
    samples.resolve_sampling_config(old)
    assert {k: getattr(old, k) for k in LEGACY_SAMPLE_CONFIG} == LEGACY_SAMPLE_CONFIG
    with pytest.raises(ValueError, match="latest_days=null"):
        samples.resolve_sampling_config(
            SimpleNamespace(sampling_config=None, reingest_from=None, latest_days=3)
        )


def test_two_distinct_slots_suffice_without_near_target_observation():
    inputs = {f"{key}_{i}": None for key in samples.BANDS for i in range(4)}
    assert not samples.eligible_inputs(inputs, T.timestamp() * 1000, CENTERED_CONFIG)
    inputs["s2_0"] = scene(-12)
    inputs["s1_0"] = scene(-11, "s1")
    # Multiple sensors in one slot do not satisfy two distinct slots.
    assert not samples.eligible_inputs(inputs, T.timestamp() * 1000, CENTERED_CONFIG)
    inputs["s2_1"] = scene(-4)
    # No observation within three days of t is required.
    assert samples.eligible_inputs(inputs, T.timestamp() * 1000, CENTERED_CONFIG)
    # Both selected slots may be before t; having data on each side is not a gate.
    inputs["s2_1"] = None
    inputs["s1_3"] = scene(12, "s1")
    assert samples.eligible_inputs(inputs, T.timestamp() * 1000, CENTERED_CONFIG)


def test_selection_cross_year_boundaries_quality_and_nearest_anchor_orbit(monkeypatch):
    pool = dict(
        s1=[scene(d, "s1", orbit=o) for d, o in [(-12, 12), (-1, 12), (10, 12), (9, 99)]],
        s2=[scene(d) for d in (-16, -9, -8, -2, 0, 7, 8, 15, 16)]
           + [scene(-0.1, fraction=0.2)],
        landsat=[],
    )

    def choose(ee, key, sample, start, end, target, minimum, orbit=None, *, centered=False):
        assert centered
        a = datetime.fromisoformat(start).replace(tzinfo=UTC).timestamp() * 1000
        b = datetime.fromisoformat(end).replace(tzinfo=UTC).timestamp() * 1000
        candidates = [
            s for s in pool[key]
            if a <= s["system:time_start"] < b and s["valid_fraction"] >= minimum
            and (orbit is None or s["relativeOrbitNumber_start"] == orbit)
        ]
        return min(candidates, key=lambda s: abs(s["system:time_start"] - T.timestamp()*1000),
                   default=None)

    monkeypatch.setattr(samples, "select_scene", choose)
    inputs = samples.select_inputs(None, SimpleNamespace(**CENTERED_CONFIG), {}, TARGET)
    assert [inputs[f"s2_{i}"]["asset_id"] for i in range(4)] == [
        "s2/-9/12", "s2/-2/12", "s2/0/12", "s2/8/12",
    ]
    assert inputs["s1_3"]["asset_id"] == "s1/10/12"
    assert inputs["s1_2"] is None
    assert samples.eligible_inputs(inputs, T.timestamp()*1000, CENTERED_CONFIG)
    metadata = samples.input_time_metadata(TARGET, inputs, CENTERED_CONFIG)
    assert metadata["observation_times_utc"]["s2_2"] == "2025-01-01T00:00:00Z"
    assert metadata["observation_times_utc"]["s1_2"] is None


def test_future_timestamps_survive_raster_ingestion_and_model_metadata(tmp_path):
    inputs = {f"{key}_{i}": None for key in samples.BANDS for i in range(4)}
    for i, offset in enumerate((-12.5, -4.25, 4.75, 12.125)):
        inputs[f"s2_{i}"] = scene(offset)
    inputs["s1_2"] = scene(5.5, "s1")  # Same slot, different actual acquisition.
    sample = dict(
        sample_id="centered", target_time=TARGET, crs="EPSG:32650",
        transform=[10, 0, 400000, 0, -10, 3200000], lon=116, lat=29,
        inputs=inputs, labels=dict(dw=None, sm=None, pml=None),
        **samples.input_time_metadata(TARGET, inputs, CENTERED_CONFIG),
    )
    sample["split"] = geographic_split(sample, 42)
    names, values = [], []
    for i in range(4):
        for key, bands in samples.BANDS.items():
            available = inputs[f"{key}_{i}"] is not None
            for band in bands:
                names.append(f"{key}_{i}_{band}")
                values.append(np.full((120, 120), 100 if available else 0, "f4"))
            names.append(f"{key}_{i}_valid")
            values.append(np.full((120, 120), available, "f4"))
    for key in ("water", "soil_moisture", "et"):
        for name in (key, key + "_valid"):
            names.append(name)
            values.append(np.zeros((120, 120), "f4"))
    path = tmp_path / "sample.tif"
    with rasterio.open(
        path, "w", driver="GTiff", height=120, width=120, count=len(names),
        dtype="float32", crs=sample["crs"], transform=Affine(*sample["transform"]),
    ) as dst:
        dst.write(np.stack(values))
        dst.descriptions = names
    ip, lp = samples.ingest(path, sample, names, tmp_path)
    group = zarr.open_group(str(tmp_path / ip), mode="r")
    expected = [inputs[f"s2_{i}"]["system:time_start"] // 1000 for i in range(4)]
    assert group["observation_time"][0, :, 1].tolist() == expected
    assert group["observation_time"][0, 2, 0] != expected[2]
    assert group["observation_time"][0, :, 2].tolist() == [0] * 4
    assert not group["sensor_valid"][0, :, 2].any()
    assert dict(group.attrs)["input_temporal_config"] == CENTERED_CONFIG
    regional = {
        key: dict(key=key.rsplit("_", 1)[0], asset_id=s["asset_id"],
                  time=s["system:time_start"] // 1000) if s else None
        for key, s in inputs.items()
    }
    cache = SimpleNamespace(
        plan=dict(config={}),
        read=lambda s, _: (
            np.zeros((len(samples.BANDS[s["key"]]), 120, 120), "f2"),
            np.ones((120, 120), bool),
        ),
    )
    inference_inputs = assemble_inputs(cache, dict(selections=json.dumps(regional)))
    np.testing.assert_array_equal(
        inference_inputs["observation_time"], group["observation_time"][0],
    )
    np.testing.assert_array_equal(inference_inputs["sensor_valid"], group["sensor_valid"][0])
    meta, _ = prepare_temporal_metadata(
        torch.from_numpy(group["sensor_valid"][:]), torch.from_numpy(group["observation_time"][:]),
    )
    assert [pair[0].replace(tzinfo=UTC).timestamp() for pair in meta[0][1].timestamps] == expected
    # A completed new dataset must not deduplicate a historical sample into itself.
    row = dict(sample, input_shard=ip, input_row=0, label_shard=lp, label_row=0)
    for key in ("inputs", "labels", "transform", "input_temporal_config",
                "slot_intervals", "observation_times_utc"):
        row[key] = json.dumps(row[key])
    new_manifest = tmp_path / "samples.parquet"
    pd.DataFrame([row]).to_parquet(new_manifest, index=False)
    old = {k: v for k, v in row.items() if k not in {
        "input_temporal_config", "temporal_sampling", "slot_intervals", "observation_times_utc",
    }}
    old_manifest = tmp_path / "old.parquet"
    pd.DataFrame([old]).to_parquet(old_manifest, index=False)
    with pytest.raises(ValueError, match="different temporal sampling"):
        merge([new_manifest, old_manifest], tmp_path / "mixed")
    assert not (tmp_path / "mixed").exists()


def test_reingestion_cannot_relabel_past_inputs_as_centered(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(LEGACY_SAMPLE_CONFIG))
    with pytest.raises(ValueError, match="cannot change observation dates"):
        samples.resolve_sampling_config(SimpleNamespace(
            sampling_config=Path("configs/experiments/legacy/data/sampling/centered8d.json"),
            reingest_from=[tmp_path],
        ))


def test_cli_refuses_new_policy_in_old_directory_before_network(tmp_path, monkeypatch):
    import sys

    def no_network(**kwargs):
        raise RuntimeError("offline initialization sentinel")

    monkeypatch.setitem(sys.modules, "ee", SimpleNamespace(Initialize=no_network))
    base = ["prepare_gee_samples.py", "--project", "test", "--output", str(tmp_path), "--plan-only"]
    legacy = base + ["--sampling-config", "configs/experiments/legacy/data/sampling/past16d.json"]
    monkeypatch.setattr(sys, "argv", legacy)
    with pytest.raises(RuntimeError, match="offline initialization sentinel"):
        samples.main()
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text())
    # Simulate a pre-versioning historical run.
    config.pop("temporal_sampling")
    config.pop("min_active_slots")
    config_path.write_text(json.dumps(config))
    original = config_path.read_bytes()
    monkeypatch.setattr(sys, "argv", base)
    with pytest.raises(RuntimeError, match="Configuration changed"):
        samples.main()
    assert config_path.read_bytes() == original
    monkeypatch.setattr(sys, "argv", legacy)
    with pytest.raises(RuntimeError, match="offline initialization sentinel"):
        samples.main()


def test_training_and_regional_centered_presets_agree():
    training = json.loads(Path("configs/experiments/legacy/data/sampling/centered8d.json").read_text())
    regional = json.loads(Path("configs/experiments/legacy/inference/poyang_centered8d.json").read_text())
    for key in ("temporal_sampling", "min_active_slots", "latest_days"):
        assert training[key] == regional[key]
    assert training["min_valid"] == regional["min_valid_fraction"] == 0.5
    assert training["min_active_slots"] == 2


@pytest.mark.parametrize("active_slots", [(), (2,), (0, 1), (2, 3), (0, 3), (0, 1, 2, 3)])
def test_training_and_regional_selection_agree_on_missing_slots(active_slots, monkeypatch):
    from hydrostate.data.gee.selection import select_window

    candidates = [scene((-12, -4, 4, 12)[i]) for i in active_slots]

    def choose(ee, key, sample, start, end, target, minimum, orbit=None, *, centered=False):
        assert centered
        if key != "s2":
            return None
        a = datetime.fromisoformat(start).replace(tzinfo=UTC).timestamp() * 1000
        b = datetime.fromisoformat(end).replace(tzinfo=UTC).timestamp() * 1000
        return next((s for s in candidates if a <= s["system:time_start"] < b), None)

    monkeypatch.setattr(samples, "select_scene", choose)
    selected = samples.select_inputs(None, SimpleNamespace(**CENTERED_CONFIG), {}, TARGET)
    cfg = dict(CENTERED_CONFIG)
    cfg["min_valid_fraction"] = cfg.pop("min_valid")
    plan = dict(
        config=cfg, training=[],
        grid=dict(crs="EPSG:32650", left=400000, top=3200000, height=32, width=32, halo=4),
    )
    regional_scenes = dict(s1=[], landsat=[], s2=[
        dict(key="s2", asset_id=s["asset_id"], time=s["system:time_start"] // 1000,
             bounds=[399000, 3190000, 410000, 3210000]) for s in candidates
    ])
    row = select_window(
        plan, dict(time=TARGET, start=TARGET, end="2025-01-09"), dict(row=0, col=0),
        regional_scenes, SimpleNamespace(read_valid=lambda *_: np.ones((120, 120), bool)),
    )
    expected = len(active_slots) >= 2
    assert samples.eligible_inputs(selected, T.timestamp()*1000, CENTERED_CONFIG) == expected
    assert row["accepted"] == expected
    inferred = json.loads(row["selections"])
    assert {key: s["asset_id"] if s else None for key, s in selected.items()} == {
        key: s["asset_id"] if s else None for key, s in inferred.items()
    }


@pytest.mark.parametrize("empty", [False, True])
def test_centered_cloud_selection_uses_aligned_full_window_and_deterministic_ties(
    monkeypatch, empty,
):
    from unittest.mock import MagicMock

    ee, collection, im, valid = (MagicMock() for _ in range(4))
    for method in ("filterBounds", "filterDate", "filter", "sort"):
        getattr(collection, method).return_value = collection
    collection.map.side_effect = lambda score: (score(im), collection)[1]
    monkeypatch.setattr(samples, "collection", lambda *_: collection)
    prepared = MagicMock()
    prepared.mask.return_value.reduce.return_value = valid
    valid.unmask.return_value = valid
    valid.gt.return_value = valid
    monkeypatch.setattr(samples, "prepare", lambda *_: prepared)
    candidates = [scene(-4), dict(scene(4), asset_id="s2/z"), dict(scene(4), asset_id="s2/a")]
    if empty:
        candidates = []

    class Count(int):
        def max(self, lower):
            return max(self, lower)

    collection.size.return_value = Count(len(candidates))
    def to_list(count):
        if count <= 0:
            raise ValueError("Collection.toList count must be positive")
        return MagicMock()

    collection.toList.side_effect = to_list
    ee.FeatureCollection.return_value.getInfo.return_value = dict(
        features=[dict(properties=s) for s in candidates],
    )
    window = dict(crs="EPSG:32650", transform=[10, 0, 400000, 0, -10, 3200000])
    selected = samples.select_scene(
        ee, "s2", window, "2024-12-16", "2025-01-17", TARGET, 0.5, centered=True,
    )
    if empty:
        assert selected is None
    else:
        assert selected["asset_id"] == "s2/a"  # Positive offset wins tie, then asset ID.
    valid.unmask.assert_called_once_with(0, sameFootprint=False)
    grid = valid.reduceRegion.call_args.kwargs
    assert grid["crs"] == window["crs"]
    assert grid["crsTransform"] == [30, 0, 400000, 0, -30, 3200000]
    assert "scale" not in grid
