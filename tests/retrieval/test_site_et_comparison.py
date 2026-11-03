import json
from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest
import zarr

from hydrostate.data.gee.contracts import fingerprint
from hydrostate.data.retrieval.specs import LABEL_SPECS, SOURCES
from hydrostate.data.site.et_comparison import (
    LocalPML,
    pair_et,
    period_end,
    pml_observation,
)
from hydrostate.data.site.et_points import select_et_windows
from hydrostate.data.site.temporal import epoch


def record(**kwargs):
    return dict(
        site_id="flux:x",
        group_id="flux:x",
        split="train",
        lon=0.006,
        lat=0.006,
        target_time="2021-01-01",
        target_end="2021-01-09",
        frequency_bin=0,
        et=dict(valid=True, value=2),
        et_corrected=dict(valid=True, value=3),
        source_file={},
        time_basis="local_standard",
        utc_offset_hours=-6,
        **kwargs,
    )


def test_missing_pml_land_et_never_uses_ew_or_corrected_station_value():
    r = record()
    paired = pair_et(r, dict(ET=0, Ew=None))
    assert paired["product_value"] == 0 and paired["site_value"] == 2
    assert paired["et_corrected"] == 3 and not paired["supervision_ready"]
    missing = pair_et(r, dict(ET=None, Ew=4))
    assert missing["product_value"] is None and missing["site_value"] is None
    assert missing["station_et"] == 2
    with pytest.raises(ValueError, match="temporal supports"):
        pair_et(r, dict(ET=1, start="2021-01-02", end="2021-01-10"))
    with pytest.raises(ValueError, match="qualified training"):
        pair_et(dict(r, split="test"), dict(ET=1))


def test_short_year_end_period_and_sparse_dates():
    assert period_end("2021-12-27") == "2022-01-01"
    assert period_end("2020-12-26") == "2021-01-01"
    rows = [
        dict(record(), target_time=f"2021-{month:02d}-{day:02d}")
        for month in (1, 2, 3)
        for day in (1, 9, 17, 25)
    ]
    rows.append(dict(record(), target_time="2021-10-01", et=dict(valid=False, value=100)))
    selected = select_et_windows(rows, 10)
    times = sorted(epoch(r["target_time"]) for r in selected)
    assert len(selected) > 1 and np.diff(times).min() >= 30 * 86400
    assert all(r["et"]["valid"] for r in selected)
    assert selected == select_et_windows(rows, 10)


def test_pml_point_cache_can_replay_without_ee(tmp_path):
    r, spec = record(), LABEL_SPECS["et_land"]
    identity = dict(
        version="site_pml_point_v1",
        sources=[asdict(s) for s in SOURCES["pml"]],
        scale=spec.scale,
        offset=spec.offset,
        lon=r["lon"],
        lat=r["lat"],
        start=r["target_time"],
        end=r["target_end"],
    )
    (tmp_path / f"{fingerprint(identity)}.json").write_text(
        json.dumps(dict(identity=identity, ET=2))
    )
    assert pml_observation(None, r, tmp_path)["ET"] == 2


def test_local_pml_reuses_exact_native_cell_and_honours_mask(tmp_path):
    projection = dict(crs="EPSG:4326", transform=[0.004, 0, 0, 0, -0.004, 0.012])
    scene = dict(
        asset_id=SOURCES["pml"][0].collection + "/2021-01-01",
        **{"system:time_start": epoch("2021-01-01") * 1000},
    )
    cache = tmp_path / "cache/pml_projection"
    cache.mkdir(parents=True)
    (cache / f"{fingerprint(scene['asset_id'])}.json").write_text(json.dumps(projection))
    sample = dict(
        lon=0.006, lat=0.006, crs="EPSG:4326", transform=[0.0001, 0, 0, 0, -0.0001, 0.012]
    )
    labels = zarr.open_group(str(tmp_path / "labels.zarr"), mode="w")
    labels.attrs.update(
        dict(
            sample=sample,
            label_sources=dict(pml=scene),
            specifications=dict(labels=dict(et_land=asdict(LABEL_SPECS["et_land"]))),
        )
    )
    labels.create_dataset("et_land_native_valid", data=np.ones((1, 9), np.uint8))
    labels.create_dataset("et_land_native_value", data=np.arange(9, dtype=np.float32)[None])
    pd.DataFrame(
        [
            dict(
                lon=0.006,
                lat=0.006,
                target_time="2021-01-01",
                target_end="2021-01-09",
                label_path="labels.zarr",
                shard_index=0,
            )
        ]
    ).to_parquet(tmp_path / "samples.parquet")
    lookup = LocalPML(tmp_path / "samples.parquet")
    assert lookup.find(record(), {})["ET"] == 4  # Middle native cell, not the nearest tile mean.
    assert lookup.find(dict(record(), lon=0.010), {})["ET"] == 5
    labels["et_land_native_valid"][0, 4] = 0
    assert lookup.find(record(), {}) is None
    assert lookup.find(dict(record(), target_end="2021-01-10"), {}) is None
