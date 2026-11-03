import numpy as np
import pytest

from hydrostate.data.site.temporal import (
    DAY,
    epoch,
    interval_et,
    morning_sm,
    windows,
)


def test_morning_utc_boundary_and_nearest_good_not_daily_mean():
    start = epoch("2021-01-01")
    # At 120 E solar 06 is previous UTC day 22; eight ideals within this UTC window.
    times = start + np.arange(8) * DAY + 22 * 3600
    result = morning_sm(
        np.r_[times - 1800, times + 1200, times + 600],
        np.r_[np.full(8, 0.1), np.full(8, 0.3), np.full(8, 0.9)],
        np.r_[np.ones(16, bool), np.zeros(8, bool)],
        120,
        "2021-01-01",
        "2021-01-09",
    )
    assert result["value"] == pytest.approx(0.3)
    assert result["valid_days"] == result["expected_days"] == 8
    assert result["morning"][0]["solar_date"] == "2021-01-02"
    assert result["morning"][-1]["solar_date"] == "2021-01-09"
    assert (
        morning_sm(times[:6], np.full(6, 0.2), np.ones(6, bool), 120, "2021-01-01", "2021-01-09")[
            "valid"
        ]
        is False
    )


def test_et_overlap_weights_qc_and_denominator():
    start = epoch("2021-06-01")
    times = np.arange(start - 1800, start + 8 * DAY, 3600)
    result = interval_et(
        times,
        times + 3600,
        np.full(len(times), 100),
        np.zeros(len(times)),
        0,
        0,
        "2021-06-01",
        "2021-06-09",
    )
    assert result["value"] == pytest.approx(100 * DAY / 2450000)
    assert result["valid_fraction"] == result["measured_fraction"] == 1
    assert result["valid"]
    filled = interval_et(
        times,
        times + 3600,
        np.full(len(times), 100),
        np.ones(len(times)),
        0,
        0,
        "2021-06-01",
        "2021-06-09",
    )
    assert filled["measured_fraction"] == 0
    assert not filled["valid"]
    with pytest.raises(ValueError, match="nonoverlapping"):
        interval_et(
            times,
            times + 4000,
            np.ones(len(times)),
            np.zeros(len(times)),
            0,
            0,
            "2021-06-01",
            "2021-06-09",
        )


def test_night_missing_is_not_good_daytime_et():
    start = epoch("2021-03-01")
    times = start + np.arange(8 * 48) * 1800
    # Only UTC daytime at Greenwich: measured >50%, but missing nights must fail.
    good = ((times % DAY) >= 5 * 3600) & ((times % DAY) < 19 * 3600)
    result = interval_et(
        times,
        times + 1800,
        np.full(len(times), 100),
        np.where(good, 0, 2),
        0,
        0,
        "2021-03-01",
        "2021-03-09",
    )
    assert "night_coverage" in result["reasons"]
    assert result["measured_fraction"] > 0.5


def test_short_year_end_window_and_future_context():
    assert list(windows("2020-12-25", "2021-01-01", as_of="2021-02-01")) == [
        ("2020-12-26", "2021-01-01")
    ]
    assert list(windows("2020-12-25", "2021-01-01", as_of="2021-01-10")) == []


def test_polar_day_has_no_required_night():
    times = epoch("2021-06-01") + np.arange(8 * 24) * 3600
    result = interval_et(
        times,
        times + 3600,
        np.full(len(times), 100),
        np.zeros(len(times)),
        90,
        0,
        "2021-06-01",
        "2021-06-09",
    )
    assert result["night_fraction"] is None
    assert result["valid"]
