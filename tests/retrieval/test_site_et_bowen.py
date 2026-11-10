"""New ET semantics: strict priority, one native-grid-aware Zarr and atomic activation."""

import json

import numpy as np
import pandas as pd
import pytest
import zarr

from hydrostate.data.gee.contracts import fingerprint
from hydrostate.data.retrieval.alignment import grid_metadata
from hydrostate.data.retrieval.contract import PROFILE
from hydrostate.data.site.active import active_release
from hydrostate.data.site.contract import VERSION, SiteSchema
from hydrostate.data.site.et_bowen import PriorityPicker
from hydrostate.data.site.et_bowen_release import pack
from hydrostate.datasets.site_dataset import SiteDataset
from hydrostate.storage import SampleShardWriter


def test_expansion_reuses_shards_and_preserves_existing_release(tmp_path):
    from hydrostate.data.site.et_bowen import seed_expansion

    source, target, scr = [tmp_path / name for name in ("old", "expanded", "scr")]
    (source / "plan").mkdir(parents=True)
    (source / "records").mkdir()
    scr.mkdir()
    (source / "publication_result.json").write_text("{}")
    (source / "collection.json").write_text('{"unchanged": true}')
    (source / "plan/previous_et_training.json").write_text('["old-train"]')
    pd.DataFrame({"candidate_id": ["a"]}).to_parquet(source / "plan/reusable.parquet")
    record = dict(candidate_id="a", input_path="zarr/a/input", label_path="zarr/a/label")
    (source / "records/a.json").write_text(json.dumps(record))
    ledger = '{"candidate_id":"a","status":"success"}\n'
    (source / "attempts.jsonl").write_text(ledger)
    pd.DataFrame([dict(valid_et=1, split="train", physical_group_id="new-train")]).to_parquet(
        scr / "inventory.parquet"
    )
    seed_expansion(source, target, scr)
    copied = json.loads((target / "records/a.json").read_text())
    assert copied["input_path"] == str(source / record["input_path"])
    assert json.loads((source / "records/a.json").read_text()) == record
    assert json.loads((target / "plan/previous_et_training.json").read_text()) == [
        "new-train",
        "old-train",
    ]
    (target / "attempts.jsonl").write_text(ledger + '{"candidate_id":"b","status":"rejected"}\n')
    seed_expansion(source, target, scr)
    assert '"candidate_id":"b"' in (target / "attempts.jsonl").read_text()


@pytest.mark.parametrize("wet_count", [0, 1])
def test_collect_keeps_all_dry_successes_even_when_wet_group_is_scarce(
    tmp_path, monkeypatch, wet_count
):
    from hydrostate.data.site import et_bowen as module

    (tmp_path / "plan").mkdir()
    (tmp_path / "plan/footprints.json").write_text('{"rows": []}')
    (tmp_path / "smoke.json").write_text('{"status": "passed"}')
    rows = [
        dict(
            candidate_id=str(i),
            frequency_bin=int(i >= 5),
            landcover=10,
            climate_class=1,
            year=2020,
            month=1,
            physical_group_id=f"s{i}",
            target_time=f"2020-01-{i + 1:02}",
        )
        for i in range(5 + wet_count)
    ]
    monkeypatch.setattr(module, "InputReuse", lambda *args: None)
    monkeypatch.setattr(module, "observation", lambda *args: ({"member": {}}, None))
    monkeypatch.setattr(
        module,
        "candidate_sample",
        lambda ee, row, *args, **kwargs: dict(
            row, status="success", label_version=module.ET_VERSION
        ),
    )
    selected = module.collect(None, pd.DataFrame(rows), tmp_path, {"seed": 1}, {}, workers=2)
    assert len(selected) == 5 + wet_count
    assert selected.frequency_bin.eq(0).sum() == 5
    assert selected.candidate_id.is_unique


def test_landcover_precedes_climate_and_failure_does_not_fill_quota():
    rows = []
    for land, climates in [(10, [1, 2, 3]), (20, [4])]:
        for climate in climates:
            for day in range(10):
                rows.append(
                    dict(
                        candidate_id=f"{land}-{climate}-{day}",
                        landcover=land,
                        climate_class=climate,
                        year=2020,
                        month=1,
                        frequency_bin=1,
                        physical_group_id=f"s{land}-{climate}",
                        target_time=f"2020-01-{day + 1:02}",
                    )
                )
    picker = PriorityPicker(rows, seed=43)
    first = picker.next({1: 10})
    picker.record(first, False)
    assert picker.counts[1] == 0
    accepted = []
    while (row := picker.next({1: 10})) is not None:
        assert row["candidate_id"] != first["candidate_id"]
        accepted.append(row)
        picker.record(row, True)
    assert pd.Series(r["landcover"] for r in accepted).value_counts().to_dict() == {10: 5, 20: 5}
    assert len(accepted) == 10


def test_active_pointer_requires_completed_child(tmp_path):
    assert active_release(tmp_path) == tmp_path
    (tmp_path / "ACTIVE.json").write_text(
        json.dumps(dict(version="site_active_release_v1", release="../x"))
    )
    with pytest.raises(ValueError, match="child"):
        active_release(tmp_path)
    (tmp_path / "ACTIVE.json").write_text(
        json.dumps(dict(version="site_active_release_v1", release="releases/new"))
    )
    with pytest.raises(ValueError, match="completed"):
        active_release(tmp_path)
    target = tmp_path / "releases/new"
    target.mkdir(parents=True)
    (target / "_PUBLISHED.json").write_text("{}")
    assert active_release(tmp_path) == target


def test_single_et_zarr_keeps_each_grid_quality_and_raw_label(tmp_path):
    config = dict(
        version=VERSION,
        input_profile=PROFILE,
        label_version="bowen_test",
        normalization_hash="norm",
    )
    (tmp_path / "collection.json").write_text(json.dumps(config))
    digest = fingerprint({k: v for k, v in config.items() if k not in ("version", "input_profile")})
    schema = SiteSchema()
    rows = []
    for i in range(6):
        grid = dict(crs="EPSG:32635", transform=[10, 0, 100 + i * 1200, 0, -10, 100])
        inputs = {k: np.zeros(s.tail_shape, s.dtype) for k, s in schema.inputs.items()}
        for k in inputs:
            if k.endswith("_valid"):
                inputs[k][...] = 1
        labels = {k: np.zeros(s.tail_shape, s.dtype) for k, s in schema.targets.items()}
        labels["site_et_native_value"][0] = 1 + i
        labels["site_et_native_valid"][0] = 1
        labels["site_et_native_weights"][0, 17:24, 17:24] = 1 / 49
        writer = SampleShardWriter(
            str(tmp_path / f"{i}/inputs.zarr"),
            str(tmp_path / f"{i}/labels.zarr"),
            1,
            schema,
            attrs=dict(
                config_sha256=digest,
                input_grids=grid_metadata(grid),
                scenes={},
                observation_records=[dict(label=dict(et=dict(value=1 + i)), spatial_quality={})],
            ),
        )
        writer.write(inputs, labels)
        writer.finalize()
        rows.append(
            dict(
                candidate_id=str(i),
                sample_id=str(i),
                physical_group_id=str(i),
                split="train",
                version=VERSION,
                input_profile=PROFILE,
                input_path=f"{i}/inputs.zarr",
                label_path=f"{i}/labels.zarr",
                shard_index=0,
                observation_capacity=1,
                crs=grid["crs"],
                transform=json.dumps(grid["transform"]),
                target_time="2020-01-01",
                target_end="2020-01-09",
                sampling_group=0,
                frequency_bin=0,
                climate_class=1,
                valid_sm=0,
                valid_et=1,
                et_support="static_210m",
                et_corrected=float(i + 1),
                et_uncorrected=float((i + 1) * 0.8),
                homogeneous_fraction=0.91,
                classification_valid_fraction=0.98,
                et_valid_fraction=0.99,
                et_measured_fraction=0.7,
                et_day_fraction=0.99,
                et_night_fraction=0.99,
            )
        )
    table = pd.DataFrame(rows)
    table.to_parquet(tmp_path / "samples.parquet", index=False)
    table["split"] = ["train", "train", "validation", "validation", "test", "test"]
    output = pack(tmp_path, table)
    group = zarr.open_group(str(output / "et.zarr"), mode="r")
    assert group.attrs["task"] == "et"
    assert group["quality/homogeneous_fraction"][:] == pytest.approx(np.full(6, 0.91))
    assert group["quality/classification_valid_fraction"][:] == pytest.approx(np.full(6, 0.98))
    for split in ("train", "validation", "test"):
        dataset = SiteDataset(output / "samples.parquet", split)
        assert len(dataset) == 2
        for i in range(2):
            item = dataset[i]
            assert not item["data_samples"]["site_sm_native_valid"].any()
            assert item["data_samples"]["site_et_native_valid"].sum() == 1
    inputs = zarr.open_group(str(output / "et.zarr/inputs"), mode="a")
    inputs["input_grid_metadata"][0] = inputs["input_grid_metadata"][1]
    with pytest.raises(ValueError, match="grids disagree"):
        SiteDataset(output / "samples.parquet", "train")[0]


def test_spatial_90_counts_missing_in_full_area_denominator():
    from hydrostate.data.site.spatial import static_quality

    labels = np.full(441, -1)
    labels[:397] = 2
    observed = labels >= 0
    probability = np.where(observed, 0.8, 0.0)
    config = dict(class_confidence=0.5, homogeneous_fraction=0.9, spatial_valid_fraction=0.9)
    q = static_quality(
        labels.reshape(21, 21), probability.reshape(21, 21), observed.reshape(21, 21), "GRA", config
    )
    assert q["accepted"]
    assert q["homogeneous_fraction"] == pytest.approx(397 / 441)
    assert q["valid_fraction"] == pytest.approx(397 / 441)
    config["homogeneous_fraction"] = 0.95
    assert not static_quality(
        labels.reshape(21, 21), probability.reshape(21, 21), observed.reshape(21, 21), "GRA", config
    )["accepted"]


def test_strict_draw_hierarchy_and_frequency_independence():
    from hydrostate.data.site.strict_sampling import balanced_draw

    rows = []
    for frequency, lands in [(0, {10: 12, 20: 8}), (1, {10: 1})]:
        for land, count in lands.items():
            for i in range(count):
                rows.append(
                    dict(
                        candidate_id=f"{frequency}-{land}-{i}",
                        frequency_bin=frequency,
                        landcover=land,
                        climate_class=1 + i % 2,
                        year=2020,
                        month=1 + (i // 2) % 2,
                        physical_group_id=str(i % 3),
                    )
                )
    chosen, audit = balanced_draw(rows, seed=7)
    assert len(chosen) == len(set(chosen))
    assert audit["by_frequency"] == {0: 17, 1: 1}
    assert audit["max_sibling_difference"] <= 1
    # Check the actual selected IDs at every level, including unselected siblings.
    selected = pd.DataFrame(rows).set_index("candidate_id").loc[chosen].reset_index()
    reference = pd.DataFrame(rows)
    prefix = ["frequency_bin"]
    for level in ["landcover", "climate_class", "year", "month"]:
        for keys, original in reference.groupby(prefix):
            keys = keys if isinstance(keys, tuple) else (keys,)
            subset = selected
            for key, value in zip(prefix, keys):
                subset = subset[subset[key].eq(value)]
            counts = subset[level].value_counts().reindex(original[level].unique(), fill_value=0)
            assert counts.max() - counts.min() <= 1
        prefix.append(level)
    assert chosen == balanced_draw(list(reversed(rows)), seed=7)[0]
