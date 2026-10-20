"""Centered observations must preserve timestamps, QA, orbit and cache isolation."""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
from shapely.geometry import box, mapping

from hydrostate.data.gee.contracts import fingerprint
from hydrostate.data.gee.products import slots
from hydrostate.data.gee.selection import epoch, select_window
from hydrostate.data.regional.planning import initialize_from_template
from hydrostate.data.regional.temporal import CENTERED, policy, regional_slots
from hydrostate.inference.planning import overlap_records, windows


def plan():
    config = dict(
        temporal_sampling=CENTERED,
        latest_days=None,
        min_active_slots=2,
        min_valid_fraction=0.5,
        cache_chunk=480,
        start="2025-01-01",
        end="2025-02-01",
    )
    return dict(
        config=config,
        identity=dict(config=copy.deepcopy(config)),
        grid=dict(crs="EPSG:32650", left=400000, top=3200000, height=32, width=32, core=32, halo=4),
        geometry=mapping(box(400000, 3199040, 400960, 3200000)),
        training=[],
        periods=[dict(time="2025-01-21", start="2025-01-17", end="2025-01-25")],
    )


def test_centered_bounds_are_half_open_and_training_slots_unchanged():
    p = plan()
    assert regional_slots("2025-01-01", p["config"]) == [
        ("2024-12-16", "2024-12-24"),
        ("2024-12-24", "2025-01-01"),
        ("2025-01-01", "2025-01-09"),
        ("2025-01-09", "2025-01-17"),
    ]
    assert regional_slots("2025-01-01", {}) == slots("2025-01-01")
    with pytest.raises(ValueError, match="latest_days=null"):
        policy(dict(temporal_sampling=CENTERED, latest_days=4))


def test_centered_selection_uses_nearest_qualified_scene_and_anchor_orbit():
    p = plan()
    t = epoch(p["periods"][0]["time"])

    def scene(key, offset, orbit=12):
        return dict(
            key=key,
            time=int(t + offset * 86400),
            orbit=orbit,
            asset_id=f"{key}/{offset}/{orbit}",
            bounds=[399000, 3190000, 410000, 3210000],
        )

    scenes = dict(
        s1=[
            scene("s1", 12, 99),
            scene("s1", -0.5, 13),
            *[scene("s1", d) for d in (-12, -1, 2, 10)],
        ],
        s2=[scene("s2", d) for d in (-17, -15, -9, -7, -1, 1, 7, 9, 15, 16)],
        landsat=[],
    )
    invalid = scene("s1", -0.5, 13)["asset_id"]
    cache = SimpleNamespace(read_valid=lambda s, w: np.full((120, 120), s["asset_id"] != invalid))
    result = select_window(p, p["periods"][0], windows(p)[0], scenes, cache)
    assert result["accepted"] and result["active_slots"] == 4
    selected = json.loads(result["selections"])
    assert [selected[f"s2_{i}"]["time"] - t for i in range(4)] == [
        d * 86400 for d in (-9, -1, 1, 9)
    ]
    assert all(selected[f"s1_{i}"]["orbit"] == 12 for i in range(4))
    assert result["latest_age_days"] == -10  # Future data must not be labelled age zero.
    assert result["nearest_observation_distance_days"] == 1
    assert result["has_before_observation"] and result["has_after_observation"]
    assert json.loads(result["observation_offsets_days"])["s2_2"] == 1
    # One supported slot is still insufficient, even if it lies exactly on t.
    scenes = dict(s1=[], s2=[scene("s2", 1), scene("s2", 16)], landsat=[])
    assert not select_window(p, p["periods"][0], windows(p)[0], scenes, cache)["accepted"]


def test_overlap_uses_future_input_support_against_original_training_support():
    p = plan()
    period = p["periods"][0]
    records = [
        dict(sample_id=day, time=day, geometry=p["geometry"])
        for day in ("2025-01-04", "2025-01-05", "2025-04-09", "2025-04-10")
    ]
    result = overlap_records(p["grid"], windows(p)[0], period, records, input_offsets=(-16, 16))
    assert result["training_context_ids"] == ["2025-01-05", "2025-04-09"]


def test_template_reuses_only_contract_matched_caches(tmp_path):
    source = tmp_path / "old"
    source.mkdir()
    p = plan()
    p["config"].pop("temporal_sampling")
    p["config"]["latest_days"] = 4
    p["identity"]["config"] = copy.deepcopy(p["config"])
    p["plan_hash"] = fingerprint(p)
    original = json.dumps(p)
    (source / "plan.json").write_text(original)
    for folder in ("input_cache", "quality_cache", "catalogs"):
        (source / folder).mkdir()
    cfg = tmp_path / "centered.json"
    cfg.write_text(json.dumps(plan()["config"]))
    target = tmp_path / "centered"
    result = initialize_from_template(target, cfg, source, reuse_input_cache=True)
    assert result["config"]["temporal_sampling"] == CENTERED
    assert (target / "input_cache").resolve() == source / "input_cache"
    assert not (target / "manifests").exists() and not (target / "preparation.sqlite").exists()
    assert (source / "plan.json").read_text() == original
    changed = dict(plan()["config"], input_profile="landsat_l2_sr_v1")
    cfg.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="identical input profile"):
        initialize_from_template(tmp_path / "wrong", cfg, source, reuse_input_cache=True)


def test_direct_pixels_catalog_includes_future_year(tmp_path, monkeypatch):
    import sys

    from hydrostate.data.regional import gee_pixels

    p = plan()
    p["config"]["gee_project"] = "test"
    period = dict(time="2025-12-29")
    years = []
    monkeypatch.setitem(
        sys.modules,
        "ee",
        SimpleNamespace(
            Initialize=lambda **kw: None, data=SimpleNamespace(setDeadline=lambda _: None)
        ),
    )
    monkeypatch.setattr(
        gee_pixels, "catalog", lambda root, plan, key, year, ee: years.append(year) or []
    )
    monkeypatch.setattr(gee_pixels, "_prepare_windows", lambda *a, **k: None)
    gee_pixels.prepare_date(tmp_path, p, period)
    assert set(years) == {2025, 2026}
