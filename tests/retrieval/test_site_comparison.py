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


def test_r2_can_be_negative_and_is_not_correlation_squared():
    rows = [dict(group_id="a", site_value=x, product_value=x + 10) for x in (1, 2, 3)]
    result = metrics(rows)
    assert result["r2"] == pytest.approx(-149)
    assert result["bias"] == 10
    assert result["rmse"] == 10
    assert result["windows"] == 3 and result["sites"] == 1
