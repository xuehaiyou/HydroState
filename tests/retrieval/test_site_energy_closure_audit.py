import numpy as np
import pandas as pd

from hydrostate.data.site.energy_closure_audit import aggregate, energy_terms


def frame():
    return pd.DataFrame(
        dict(
            start=pd.to_datetime(["2020-01-01", "2022-01-01"]),
            end=pd.to_datetime(["2020-01-01 00:30", "2022-01-01 00:30"]),
            LE_F_MDS=[40.0, 80.0],
            H_F_MDS=[20.0, 40.0],
            G_F_MDS=[0.0, 0.0],
            NETRAD=[100.0, 200.0],
            SW_IN_F_MDS=[10.0, 100.0],
            LE_F_MDS_QC=[0, 0],
            H_F_MDS_QC=[0, 0],
            G_F_MDS_QC=[0, 0],
        )
    )


def test_common_mask_and_sum_ratio():
    f = frame()
    f.loc[0, "NETRAD"] = 50.0
    assert aggregate(f)["ebr"] == 180 / 250
    f.loc[0, "H_F_MDS"] = np.nan
    assert aggregate(f)["ebr"] == 0.6
    assert aggregate(f)["status"] == "insufficient_record_span"


def test_quality_and_daylight_are_separate():
    f = frame()
    f.loc[1, "LE_F_MDS_QC"] = 1
    assert energy_terms(f)[0].tolist() == [True, False]
    assert energy_terms(f, True)[0].tolist() == [True, True]
    assert aggregate(f, True, True)["valid_intervals"] == 1


def test_missing_ground_is_not_zero_and_threshold_strict():
    f = frame()
    f.loc[:, "H_F_MDS"] = [35.0, 70.0]
    assert aggregate(f)["ebr"] == 0.75
    assert aggregate(f)["status"] == "fail"
    f.loc[:, "G_F_MDS"] = np.nan
    assert aggregate(f)["ebr"] is None
