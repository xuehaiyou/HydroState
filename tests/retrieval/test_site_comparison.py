import pytest

from hydrostate.data.site.comparison import metrics, pair_sm
from hydrostate.data.site.temporal import epoch


def test_pairing_uses_common_dates_and_preserves_training_value():
    record = dict(
        site_id="a",
        group_id="a",
        target_time="2021-01-01",
        source_file={},
        sm=dict(
            value=0.6,
            morning=[
                dict(utc=epoch("2021-01-01T22:10:00"), offset_seconds=600, value=0.2),
                dict(utc=epoch("2021-01-02T22:00:00"), offset_seconds=0, value=0.8),
            ],
        ),
    )
    product = dict(
        projection={},
        observations=[
            dict(date="2021-01-01", soil_moisture_am=0.3, retrieval_qual_flag_am=0, asset_id="x"),
            dict(date="2021-01-02", soil_moisture_am=0.9, retrieval_qual_flag_am=1, asset_id="y"),
        ],
    )
    result = pair_sm(record, product)
    assert result["training_value"] == 0.6
    assert result["site_value"] == 0.2
    assert result["product_value"] == 0.3
    assert result["common_days"] == 1
    assert result["excluded_days"][0]["reason"] == "SMAP_missing_or_QA"
    assert metrics([result])["r2"] is None
    empty = pair_sm(record, dict(projection=None, observations=[]))
    assert empty["reason"] == "no_product_images_in_window"
    assert empty["site_value"] is None and empty["training_value"] == 0.6
    rejected = pair_sm(record, dict(product, observations=product["observations"][1:]))
    assert rejected["reason"] == "no_common_QA_valid_dates"


def test_r2_can_be_negative_and_is_not_correlation_squared():
    rows = [dict(group_id="a", site_value=x, product_value=x + 10) for x in (1, 2, 3)]
    result = metrics(rows)
    assert result["r2"] == pytest.approx(-149)
    assert result["bias"] == 10
    assert result["rmse"] == 10
    assert result["windows"] == 3 and result["sites"] == 1


def test_existing_point_cache_never_calls_gee(tmp_path):
    import json
    from dataclasses import asdict

    from hydrostate.data.gee.contracts import fingerprint
    from hydrostate.data.retrieval.specs import SOURCES
    from hydrostate.data.site.comparison import smap_observations

    record = dict(lon=25.0, lat=41.0, target_time="2021-01-01", target_end="2021-01-09")
    identity = dict(
        sources=[asdict(s) for s in SOURCES["sm"]],
        lon=25.0,
        lat=41.0,
        start="2021-01-01",
        end="2021-01-09",
    )
    old = tmp_path / "old"
    old.mkdir()
    data = dict(identity=identity, observations=[], projection=None)
    (old / f"{fingerprint(identity)}.json").write_text(json.dumps(data))
    # None is deliberately unusable as an EE client: a local hit must not access the network.
    result = smap_observations(None, record, tmp_path / "new", reuse_caches=[old])
    assert result["response_source"] == "existing_point_cache"


def test_local_native_smap_reuse_requires_actual_cell_and_version(tmp_path):
    import json

    import pandas as pd

    from hydrostate.data.site.comparison import LocalSMAP

    label = tmp_path / "labels.zarr"
    label.mkdir()
    pd.DataFrame(
        [dict(target_time="2021-01-01", target_end="2021-01-09", label_path="labels.zarr")]
    ).to_parquet(tmp_path / "samples.parquet")
    attrs = dict(
        sm_quality=dict(
            support=dict(
                grid=dict(crs="EPSG:4326", transform=[0.08, 0, -180, 0, -0.08, 85]),
                cell=[2250, 1062],
            ),
            observations=[
                dict(
                    date="2021-01-02",
                    soil_moisture_am=0.2,
                    asset_id="NASA/SMAP/SPL3SMP_E/005/20210102",
                )
            ],
        )
    )
    (label / ".zattrs").write_text(json.dumps(attrs))
    lookup = LocalSMAP(tmp_path / "samples.parquet")
    record = dict(target_time="2021-01-01", target_end="2021-01-09", lon=0.04, lat=0.0)
    value = lookup.find(record, {})
    assert value["observations"][0]["retrieval_qual_flag_am"] == 0
    assert lookup.find(dict(record, lon=0.14), {}) is None
    attrs["sm_quality"]["observations"][0]["asset_id"] = "NASA/SMAP/SPL3SMP_E/006/20210102"
    (label / ".zattrs").write_text(json.dumps(attrs))
    assert lookup.find(record, {}) is None
