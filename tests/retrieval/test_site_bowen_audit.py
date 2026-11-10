"""Scientific invariants of the isolated read-only corrected-label audit."""

import numpy as np
import pandas as pd
import pytest

from hydrostate.data.site.bowen_audit import daily_support, label_window, prepare, split_local_days
from hydrostate.data.site.temporal import DAY, epoch


def source(tmp_path, qc=None, corrected=None, offset=0):
    stamps = pd.date_range("2020-06-01", periods=48 * 10, freq="30min")
    n = len(stamps)
    frame = pd.DataFrame(
        dict(
            TIMESTAMP_START=stamps.strftime("%Y%m%d%H%M"),
            TIMESTAMP_END=(stamps + pd.Timedelta(minutes=30)).strftime("%Y%m%d%H%M"),
            LE_F_MDS=np.full(n, 100.0),
            LE_F_MDS_QC=np.zeros(n) if qc is None else qc,
            LE_CORR=np.full(n, 125.0) if corrected is None else corrected,
            H_F_MDS=np.full(n, 200.0),
            H_CORR=np.full(n, 250.0),
        )
    )
    path = tmp_path / "flux.csv"
    frame.to_csv(path, index=False)
    return prepare(path, offset)


def test_constant_flux_conversion_and_bowen_preservation(tmp_path):
    data, info = source(tmp_path)
    row = label_window(data, dict(lat=0, lon=0), "2020-06-02", "2020-06-10")
    assert row["corrected_valid"]
    assert row["corrected_value"] == pytest.approx(125 * DAY / 2450000)
    assert row["raw_matched_value"] == pytest.approx(100 * DAY / 2450000)
    assert info["bowen_preservation"]["relative_cross_product_p95"] == 0


def test_filled_coverage_is_not_measured_coverage(tmp_path):
    data, _ = source(tmp_path, qc=np.ones(480))
    row = label_window(data, dict(lat=0, lon=0), "2020-06-02", "2020-06-10")
    assert row["corrected_valid_fraction"] == 1
    assert row["corrected_measured_fraction"] == 0
    assert not row["corrected_valid"]
    assert row["corrected_reasons"] == "measured_duration"


def test_day_threshold_inclusive_and_missing_records_in_denominator():
    start = epoch("2020-01-01")
    stamps = start + np.arange(48) * 1800
    valid = np.arange(48) < 36
    mask, info = daily_support(stamps, stamps + 1800, valid)
    assert mask.sum() == 36 and info["accepted_days"] == 1
    valid[35] = False
    mask, info = daily_support(stamps, stamps + 1800, valid)
    assert not mask.any() and info["accepted_days"] == 0
    # Missing rows are not silently removed from the complete-day denominator.
    mask, _ = daily_support(stamps[:35], stamps[:35] + 1800, np.ones(35, bool))
    assert not mask.any()


def test_midnight_split_conserves_energy_and_offset(tmp_path):
    frame = pd.DataFrame(
        dict(TIMESTAMP_START=["202006012345"], TIMESTAMP_END=["202006020015"], LE_F_MDS=[100.0])
    )
    result = split_local_days(frame)
    assert len(result) == 2
    assert (result.local_end - result.local_start).sum() == 1800
    data, _ = source(tmp_path, offset=5.5)
    assert data["starts"][0] == epoch("2020-06-01") - 19800


def test_common_support_and_no_raw_fallback(tmp_path):
    corrected = np.full(480, 125.0)
    # One full local day absent: common support uses the other seven days.
    corrected[48:96] = np.nan
    data, _ = source(tmp_path, corrected=corrected)
    data["raw"] = data["raw"].copy()
    data["raw"][48:96] = 1000
    row = label_window(data, dict(lat=0, lon=0), "2020-06-02", "2020-06-10")
    assert row["corrected_valid"]
    assert row["corrected_valid_fraction"] == pytest.approx(7 / 8)
    assert row["raw_matched_value"] == pytest.approx(100 * DAY / 2450000)
    assert row["raw_value"] > row["raw_matched_value"]
    data["corrected_mask"][:] = False
    row = label_window(data, dict(lat=0, lon=0), "2020-06-02", "2020-06-10")
    assert not row["corrected_valid"] and row["corrected_value"] is None


def test_overlap_is_rejected():
    frame = pd.DataFrame(
        dict(
            TIMESTAMP_START=["202006010000", "202006010015"],
            TIMESTAMP_END=["202006010030", "202006010045"],
        )
    )
    with pytest.raises(ValueError, match="overlapping"):
        split_local_days(frame)
