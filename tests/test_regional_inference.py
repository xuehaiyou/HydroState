"""Regional contract tests: dates, registration, missing inputs and durable output."""

import json

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box, mapping

from hydrostate.data.gee_sampling import BANDS, MODALITIES
from hydrostate.inference.gee import SceneCache, epoch, normalize_pixels, select_window
from hydrostate.inference.planning import (
    input_shape,
    input_transform,
    make_grid,
    map_point,
    output_transform,
    overlap_records,
    periods,
    training_records,
    window_geometry,
    windows,
)
from hydrostate.inference.predict import WeightedMosaic, create_product, write_date


@pytest.fixture
def plan():
    geometry = box(400000, 3200000, 401920, 3201920)
    return dict(
        plan_hash="test-plan",
        geometry=mapping(geometry),
        grid=make_grid(geometry, "EPSG:32650"),
        config=dict(cache_chunk=480, min_active_slots=2, latest_days=3, min_valid_fraction=0.5),
        periods=[dict(time="2025-01-21", start="2025-01-17", end="2025-01-25")],
        training=[],
    )


def test_dates_keep_annual_support_and_leap_years():
    items = periods("2017-04-01", "2026-01-01")
    assert len(items) == 403
    assert items[0] == dict(time="2017-04-03", start="2017-03-30", end="2017-04-07")
    assert items[-1] == dict(time="2025-12-29", start="2025-12-27", end="2026-01-01")
    assert all("2017-04-01" <= p["time"] < "2026-01-01" for p in items)
    last_leap = [p for p in items if p["start"] == "2020-12-26"][0]
    assert last_leap["end"] == "2021-01-01"


def test_patch_centres_align_and_partition_without_gaps(plan):
    grid = plan["grid"]
    counts = np.zeros((grid["height"], grid["width"]), "i4")
    h, w = input_shape(grid)
    for window in windows(plan):
        y, x, core = window["row"], window["col"], grid["core"]
        assert map_point(input_transform(grid), x * 3 + 12, y * 3 + 12) == map_point(
            output_transform(grid), x, y
        )
        assert y * 3 + 120 <= h and x * 3 + 120 <= w
        counts[y : y + core, x : x + core] += 1
    assert (counts == 1).all()


def test_overlapping_patches_blend_and_never_fill_missing_support():
    mosaic = WeightedMosaic((32, 48), 32)
    a = {
        k: np.full((32, 32), v, "f4")
        for k, v in [("water", 0.1), ("soil_moisture", 0.2), ("et", 2.0)]
    }
    b = {
        k: np.full((32, 32), v, "f4")
        for k, v in [("water", 0.9), ("soil_moisture", 0.4), ("et", 4.0)]
    }
    a["et"][0, :16] = np.nan
    mosaic.add(0, 0, a)
    mosaic.add(0, 16, b)
    result = mosaic.finish()
    assert np.isnan(result["et"][0, :16]).all()
    assert np.max(np.diff(result["et"][16])) < 0.3
    assert np.allclose(result["et"][16, :16], 2.0)
    assert np.allclose(result["et"][16, 32:], 4.0)
    assert mosaic.disagreement()["et"]["mean_std"] > 0


def test_overlapping_grid_keeps_every_context_inside_input():
    geometry = box(400000, 3200000, 409600, 3209600)
    grid = make_grid(geometry, "EPSG:32650", core=32, stride=16)
    h, w = input_shape(grid)
    count = np.zeros((grid["height"], grid["width"]), "i4")
    for item in windows(dict(grid=grid, geometry=mapping(geometry))):
        y, x = item["row"], item["col"]
        assert y * 3 + 120 <= h and x * 3 + 120 <= w
        count[y : y + 32, x : x + 32] += 1
    assert count.min() >= 1 and count.max() == 4


def test_overlap_is_descriptive_and_checks_input_context(plan):
    window = windows(plan)[0]
    geometry = mapping(window_geometry(plan["grid"], window))
    training = [
        dict(sample_id="old", time="2020-01-01", geometry=geometry),
        dict(sample_id="context", time="2024-12-25", geometry=geometry),
        dict(sample_id="target", time="2025-01-20", geometry=geometry),
    ]
    result = overlap_records(plan["grid"], window, plan["periods"][0], training)
    assert result["training_spatial_ids"] == ["old", "context", "target"]
    assert result["training_context_ids"] == ["context", "target"]
    assert result["training_target_ids"] == ["target"]


def test_distant_training_records_cannot_project_to_infinite_overlaps(tmp_path, plan):
    manifest = tmp_path / "samples.parquet"
    pd.DataFrame(
        [dict(lon=-63.0, lat=-29.0, split="train", sample_id="far", target_time="2025-01-21")]
    ).to_parquet(manifest)
    assert (
        training_records(manifest, plan["grid"]["crs"], box(400000, 3200000, 402000, 3202000)) == []
    )


def test_normalization_preserves_training_band_units_and_mask():
    norm = {MODALITIES["s1"]: {"vv": dict(mean=-10.0, std=2.0), "vh": dict(mean=-20.0, std=4.0)}}
    values = np.array([[[-10.0, 999.0]], [[-4.0, 999.0]]], dtype="f4")
    output = normalize_pixels("s1", values, np.array([[True, False]]), norm)
    assert output.dtype == np.float16
    np.testing.assert_array_equal(output, [[[0.5, 0]], [[1.5, 0]]])


def test_scene_selection_locks_orbit_and_accepts_missing_s2(plan):
    def scene(key, day, orbit=None, fraction=1):
        return dict(
            key=key,
            time=epoch(day),
            asset_id=key + day,
            orbit=orbit,
            bounds=[399000, 3190000, 410000, 3210000],
            fraction=fraction,
        )

    scenes = dict(
        s1=[
            scene("s1", "2025-01-21", 12),
            scene("s1", "2025-01-03", 13),
            scene("s1", "2025-01-02", 12),
        ],
        s2=[],
        landsat=[],
    )

    class Cache:
        def read(self, scene, window):
            return None, np.ones((120, 120), bool)

    result = select_window(plan, plan["periods"][0], windows(plan)[0], scenes, Cache())
    assert result["accepted"]
    selected = json.loads(result["selections"])
    assert selected["s1_2"]["orbit"] == 12
    assert selected["s2_3"] is None
    assert result["active_slots"] == 2
    scenes["s1"] = scenes["s1"][1:]
    assert not select_window(plan, plan["periods"][0], windows(plan)[0], scenes, Cache())[
        "accepted"
    ]


def test_cache_reads_require_committed_chunks_and_preserve_offsets(tmp_path, plan):
    cache = SceneCache(tmp_path, plan)
    scene = dict(key="s1", asset_id="S1/test", time=epoch("2025-01-21"))
    group = cache._open(scene, writable=True)
    window = windows(plan)[0]
    with pytest.raises(FileNotFoundError):
        cache.read(scene, window)
    h, w = input_shape(plan["grid"])
    value = np.arange(h * w, dtype="f4").reshape(h, w) / 10000
    group["values"][:] = np.stack([value, value * 2])
    group["valid"][:] = 1
    group["done"][:] = 1
    actual, valid = cache.read(scene, window)
    np.testing.assert_array_equal(actual[0], value[:120, :120].astype("f2"))
    assert valid.all()


def test_zarr_masks_commit_markers_provenance_and_resume(tmp_path, plan):
    provenance = dict(checkpoint_sha256="weights", statistics_sha256="stats")
    group = create_product(tmp_path, plan, provenance)
    assert group["date_complete"][:].tolist() == [0]
    assert np.isnan(group["water"][0]).all()
    rows = []
    for window in windows(plan):
        selections = {f"{k}_{t}": None for k in BANDS for t in range(4)}
        rows.append(
            dict(
                window,
                selections=json.dumps(selections),
                accepted=False,
                training_context_ids='["sample"]',
                training_target_ids="[]",
            )
        )
    h, w = plan["grid"]["height"], plan["grid"]["width"]
    maps = {k: np.full((h, w), np.nan, "f4") for k in ("water", "soil_moisture", "et")}
    maps["water"][4, 4] = 1
    maps["et"][4, 4] = 3
    report = write_date(group, plan, 0, maps, pd.DataFrame(rows))
    assert report["accepted_windows"] == 0
    assert group["date_complete"][:].tolist() == [1]
    assert int(group["quality"][0, 4, 4]) == 1 | 4 | 8 | 64
    assert np.isnan(group["soil_moisture"][0, 4, 4])
    reopened = create_product(tmp_path, plan, provenance)
    assert reopened["date_complete"][0] == 1
    with pytest.raises(ValueError, match="different checkpoint"):
        create_product(tmp_path, plan, dict(provenance, checkpoint_sha256="other"))
