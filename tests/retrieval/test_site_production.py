import json

import numpy as np
import pandas as pd
import pytest
import zarr

from hydrostate.data.site import production, release
from hydrostate.data.site.contract import SiteSchema
from hydrostate.data.site.spatial import footprint_operator, static_quality, static_weights
from hydrostate.datasets.site_dataset import SiteDataset

CONFIG = dict(
    class_confidence=0.6,
    spatial_valid_fraction=0.95,
    homogeneous_fraction=0.95,
    footprint_min_mass=0.95,
    min_date_gap_days=30,
    seed=43,
    min_free_gib=0,
    footprint_root="unused",
    split_root="unused",
    product_manifest="unused",
)


def test_whole_area_95_percent_and_unknown_denominator():
    labels = np.ones((21, 21), dtype=int)
    confidence, observed = np.ones((21, 21)), np.ones((21, 21))
    observed.flat[:22] = 0
    assert static_quality(labels, confidence, observed, "ENF", CONFIG)["accepted"]
    observed.flat[22] = 0
    assert not static_quality(labels, confidence, observed, "ENF", CONFIG)["accepted"]
    observed[:] = 1
    confidence.flat[:23] = 0.1
    q = static_quality(labels, confidence, observed, "ENF", CONFIG)
    assert q["valid_fraction"] == 1 and not q["accepted"]
    assert not static_quality(labels, np.ones_like(confidence), observed, "GRA", CONFIG)["accepted"]


def test_absolute_footprint_mass_not_renormalized_before_gate(tmp_path):
    g = zarr.open_group(str(tmp_path / "ffp"), mode="w")
    g.attrs.update(complete=True, footprint_tile_valid=True)
    mass = np.zeros((40, 40), "float32")
    mass[20, 20], mass[20, 21] = 0.94, 0.03
    g.create_dataset("mass", data=mass)
    support = np.ones((40, 40), bool)
    weights, quality = footprint_operator(tmp_path / "ffp", support, 0.95)
    assert quality["accepted"] and weights.sum() == pytest.approx(1)
    support[20, 21] = False
    weights, quality = footprint_operator(tmp_path / "ffp", support, 0.95)
    assert not quality["accepted"] and weights.sum() == 0


def test_static_support_never_shrinks_and_sm_is_independent():
    obs = [
        dict(
            target=t,
            member=dict(candidate_id=t, operator_ready=t == "sm"),
            label={t: dict(value=0.2)},
        )
        for t in ("sm", "et")
    ]
    support = np.ones((40, 40), bool)
    support[17, 17] = False
    accepted, rejected = production.prepare_operators(
        obs, support, {"et": dict(accepted=True, operator="static_210m")}, CONFIG
    )
    assert [i["target"] for i in accepted] == ["sm"]
    assert len(rejected) == 1
    assert accepted[0]["weights"][20, 20] == 1
    assert np.count_nonzero(static_weights()) == 49


def fixture_plan(tmp_path):
    plan = tmp_path / "plan"
    plan.mkdir()
    rows, members = [], []
    for cid, bin_id, site, day in [
        ("a", 0, "a", "2021-01-01"),
        ("b", 1, "b", "2021-01-01"),
        ("c", 1, "b", "2021-01-09"),
        ("d", 0, "a", "2021-01-09"),
    ]:
        end = (pd.Timestamp(day) + pd.Timedelta(days=8)).date().isoformat()
        row = dict(
            candidate_id=cid,
            physical_group_id=site,
            spatial_group_id=site,
            site_id=site,
            split="train",
            frequency_bin=bin_id,
            lon=25.12345678912345,
            lat=41.12345678912345,
            landcover=30,
            climate_class=8,
            year=2021,
            month=1,
            target_time=day,
            target_end=end,
            draw_order=len(rows),
        )
        rows.append(row)
        label = dict(
            target_time=day,
            target_end=end,
            source="ismn",
            support_type="point",
            sm=dict(valid=True, value=0.2),
        )
        path = tmp_path / f"{cid}.json"
        path.write_text(json.dumps(dict(windows=[label])))
        members.append(
            dict(
                row,
                target="sm",
                source="ismn",
                operator_ready=True,
                label_record=str(path),
                record_index=0,
                input_candidate_id=cid,
            )
        )
    pd.DataFrame(rows[:2]).to_parquet(plan / "selected.parquet", index=False)
    pd.DataFrame(rows).to_parquet(plan / "input_candidates.parquet", index=False)
    pd.DataFrame(members).to_parquet(plan / "label_members.parquet", index=False)
    return plan


def test_resume_initial_first_same_bin_refill_and_real_zarr(tmp_path, monkeypatch):
    plan = fixture_plan(tmp_path)
    root = tmp_path / "out"
    monkeypatch.setattr(release, "lineage", lambda *a: {})
    calls = []
    schema = SiteSchema()

    def inputs(ee, sample, cache, norm, config):
        calls.append(sample["candidate_id"])
        arrays = {k: np.zeros(s.tail_shape, s.dtype) for k, s in schema.inputs.items()}
        for k in arrays:
            if k.endswith("_valid"):
                arrays[k][:] = 1
            if k.endswith("_quality"):
                arrays[k][:, 0] = 1
        arrays["target_bounds"][:] = [
            production.epoch(sample[k]) for k in ("target_time", "target_end")
        ]
        return arrays, {}, sample["candidate_id"] != "b"

    monkeypatch.setattr(production, "input_arrays", inputs)
    first = production.collect_production(None, plan, root, CONFIG, {}, workers=1, max_attempts=1)
    assert not first["complete"] and calls == ["a"]
    second = production.collect_production(None, plan, root, CONFIG, {}, workers=1)
    assert second["complete"] and calls == ["a", "b", "c"]
    assert second["successful_by_bin"] == {0: 1, 1: 1}
    table = pd.read_parquet(root / "samples.parquet")
    assert table.lon.iloc[0] == 25.12345678912345
    assert len(SiteDataset(root / "samples.parquet", "train")) == 2
    item = SiteDataset(root / "samples.parquet", "train")[0]
    assert item["data_samples"]["site_sm_native_weights"][0, 20, 20] == 1
    audit = release.audit_production(root, plan, require_complete=True)
    assert audit["samples"] == 2
    production.collect_production(None, plan, root, CONFIG, {}, workers=1)
    assert calls == ["a", "b", "c"]


def test_known_failed_footprint_is_found_across_source_name(tmp_path):
    p = tmp_path / "stations" / "BE-Bra" / "windows.json"
    p.parent.mkdir(parents=True)
    row = dict(target_time="2021-01-01", target_end="2021-01-09", footprint_tile_valid=False)
    p.write_text(json.dumps(dict(windows=[row])))
    assert production.known_footprint(dict(row, site_id="fluxnet:BE-Bra"), tmp_path) == row
