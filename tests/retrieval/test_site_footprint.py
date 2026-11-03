import numpy as np
import pandas as pd
import pytest
import zarr

from hydrostate.data.site.archives import sha256
from hydrostate.data.site.availability import availability_rows
from hydrostate.data.site.footprint import (
    FFP_INTEGRAL,
    density,
    integration_grid,
    parameter_valid,
    window_mass,
)
from hydrostate.data.site.icos_footprints import (
    history_values,
    metadata_events,
    process_station,
    read_intervals,
)
from hydrostate.data.site.temporal import DAY, epoch, interval_et

PARAMS = dict(zm=10.0, ws=3.0, h=1000.0, ol=-100.0, sigmav=0.8, ustar=0.4, wd=0.0)


def test_author_v142_unsmoothed_reference_values():
    # Produced with the author's unmodified FFP_climatology, smooth_data=None.
    east, north = np.array([100.0, 150.0, 300.0, 300.0]), np.array([150.0, 50.0, 0.0, 200.0])
    expected = [
        1.170607369043935e-5,
        3.211858445900956e-7,
        3.776245632193192e-15,
        1.135044602795495e-7,
    ]
    actual = density(east, north, **dict(PARAMS, wd=35.0)) * FFP_INTEGRAL
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-22)


def test_from_wind_rotation_and_parameter_domain():
    east, north = np.meshgrid(np.arange(-300, 301, 10.0), np.arange(-300, 301, 10.0))
    north_wind = density(east, north, **PARAMS)
    east_wind = density(east, north, **dict(PARAMS, wd=90.0))
    assert north_wind[north <= 0].sum() == 0
    assert east_wind[east < 0].sum() == 0
    np.testing.assert_allclose(north_wind.T, east_wind, atol=1e-16)
    assert parameter_valid(canopy=0.2, **PARAMS)
    assert not parameter_valid(canopy=10, **PARAMS)  # Roughness sublayer
    for key, value in [("ustar", 0.1), ("ol", 0.0), ("h", 10.0), ("wd", -1.0), ("sigmav", np.nan)]:
        assert not parameter_valid(canopy=0.2, **dict(PARAMS, **{key: value}))


def test_tile_mass_not_renormalized_and_interval_duration_weighting():
    east, north, area = integration_grid(9.0, 45.0, 5.0)
    expected = density(east, north, **PARAMS) * area
    expected = expected.reshape(40, 6, 40, 6).sum(axis=(1, 3))
    result, quality = window_mass(
        [PARAMS, dict(PARAMS, wd=180)], [1.0, 3.0], 9.0, 45.0, steps=(10.0, 5.0)
    )
    south = density(east, north, **dict(PARAMS, wd=180)) * area
    np.testing.assert_allclose(
        result,
        (expected + 3 * south.reshape(40, 6, 40, 6).sum(axis=(1, 3))) / 4,
        rtol=1e-6,
        atol=1e-9,
    )
    assert result.sum() < 0.95  # Its tail is outside this tile; no normalization hides it.
    assert not quality["footprint_tile_valid"]


def test_metadata_priority_no_future_backfill_and_expiry(tmp_path):
    path = tmp_path / "metadata.csv"
    rows = []
    for group, values in enumerate(
        [
            dict(HEIGHTC="1", HEIGHTC_DATE="20210101", HEIGHTC_STATISTIC="Mean"),
            dict(HEIGHTC="2", HEIGHTC_DATE="20210101", HEIGHTC_STATISTIC="90th Percentile"),
            dict(
                HEIGHTC="99",
                HEIGHTC_DATE="20210101",
                HEIGHTC_STATISTIC="Mean",
                HEIGHTC_SPP="species",
            ),
            dict(
                HEIGHTC="3",
                HEIGHTC_DATE_START="20210201",
                HEIGHTC_DATE_END="20210228",
                HEIGHTC_STATISTIC="Mean",
            ),
        ]
    ):
        rows.extend(
            dict(GROUP_ID=group, VARIABLE_GROUP="GRP_HEIGHTC", VARIABLE=k, DATAVALUE=v)
            for k, v in values.items()
        )
    pd.DataFrame(rows).to_csv(path, index=False)
    events, _ = metadata_events(path, "canopy", 1)
    assert events == [(epoch("2021-01-01") - 3600, 2.0), (epoch("2021-02-28") - 3600, 3.0)]
    values = history_values(
        events,
        np.array([epoch("2020-12-01"), epoch("2021-01-02"), epoch("2021-02-05")]),
        max_age_days=30,
    )
    assert np.isnan(values[0]) and values[1] == 2 and np.isnan(values[2])


def test_metadata_conflict_is_local_to_period_not_entire_station(tmp_path):
    path = tmp_path / "metadata.csv"
    rows = []
    for group, (day, statistic, height) in enumerate(
        [
            ("20210101", "Mean", 1),
            ("20210101", "Mean", 2),
            ("20210101", "90th Percentile", 3),
            ("20210201", "90th Percentile", 4),
            ("20210201", "90th Percentile", 5),
            ("20210301", "Mean", 6),
        ]
    ):
        values = dict(HEIGHTC=str(height), HEIGHTC_DATE=day, HEIGHTC_STATISTIC=statistic)
        rows.extend(
            dict(GROUP_ID=group, VARIABLE_GROUP="GRP_HEIGHTC", VARIABLE=k, DATAVALUE=v)
            for k, v in values.items()
        )
    pd.DataFrame(rows).to_csv(path, index=False)
    events, reasons = metadata_events(path, "canopy", 0)
    result = history_values(
        events, np.array([epoch("2021-01-02"), epoch("2021-02-02"), epoch("2021-03-02")])
    )
    assert result[0] == 3 and np.isnan(result[1]) and result[2] == 6
    assert reasons["ambiguous_height_update"] == 1


def test_local_standard_to_utc_and_common_support_rechecks_nights(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("TIMESTAMP_START,TIMESTAMP_END,LE\n202101010000,202101010030,100\n")
    frame = read_intervals(path, ["LE"], 1)
    assert frame.TIMESTAMP_START.iloc[0] == epoch("2020-12-31T23:00:00")
    start = epoch("2021-03-01")
    times = start + np.arange(8 * 48) * 1800
    daytime = (times % DAY >= 6 * 3600) & (times % DAY < 18 * 3600)
    args = (
        times,
        times + 1800,
        np.full(len(times), 100.0),
        np.zeros(len(times)),
        0,
        0,
        "2021-03-01",
        "2021-03-09",
    )
    assert interval_et(*args)["valid"]
    result = interval_et(*args, support_valid=daytime)
    assert not result["valid"] and "night_coverage" in result["reasons"]


def test_availability_deduplicates_sensors_and_preserves_empty_groups():
    table = pd.DataFrame(
        [
            dict(physical_group_id="a", target_time=t, split="train", frequency_bin=1)
            for t in ["2021-01-01", "2021-01-01", "2021-01-09", "2021-02-02"]
        ]
    )
    rows = availability_rows(table, "SM")
    selected = next(r for r in rows if r["split"] == "train" and r["frequency_bin"] == 1)
    assert selected["station_windows"] == 3 and selected["capacity_30d"] == 2
    assert all(r["station_windows"] == 0 for r in rows if r["frequency_bin"] != 1)


def test_reject_invalid_durations():
    with pytest.raises(ValueError, match="durations"):
        window_mass([PARAMS], [-1], 9.0, 45.0)


def test_real_et_support_zarr_and_resume(tmp_path):
    prefix = tmp_path / "ICOSETC_ZZ-Test"
    times = pd.date_range("2021-12-27", "2022-01-01", freq="h", inclusive="left")
    frame = pd.DataFrame(
        dict(
            TIMESTAMP_START=times.strftime("%Y%m%d%H%M"),
            TIMESTAMP_END=(times + pd.Timedelta(hours=1)).strftime("%Y%m%d%H%M"),
        )
    )
    source = tmp_path / "ICOSETC_ZZ-Test_FLUXNET_HH_L2.csv"
    frame.assign(LE_F_MDS=100.0, LE_F_MDS_QC=0).to_csv(source, index=False)
    frame.assign(
        WS=3.0, PBLH=1000.0, MO_LENGTH=-100.0, V_SIGMA=1.0, USTAR=0.5, WD=180.0, FOOTPRINT_FLAG=3
    ).to_csv(f"{prefix}_FLUXES_L2.csv", index=False)
    for suffix, group, values in [
        (
            "VARINFO_FLUXES",
            "GRP_VAR_INFO",
            dict(VAR_INFO_VARNAME="USTAR", VAR_INFO_HEIGHT="5.2", VAR_INFO_DATE="20210101"),
        ),
        (
            "ANCILLARY",
            "GRP_HEIGHTC",
            dict(HEIGHTC=".3", HEIGHTC_DATE="20211201", HEIGHTC_STATISTIC="Mean"),
        ),
        ("SITEINFO", "GRP_UTC_OFFSET", dict(UTC_OFFSET="0")),
    ]:
        pd.DataFrame(
            [
                dict(GROUP_ID=1, VARIABLE_GROUP=group, VARIABLE=k, DATAVALUE=v)
                for k, v in values.items()
            ]
        ).to_csv(f"{prefix}_{suffix}_L2.csv", index=False)
    row = dict(
        site_id="icos:ZZ-Test",
        source_path=str(source),
        source_bytes=source.stat().st_size,
        source_mtime_ns=source.stat().st_mtime_ns,
        utc_offset_hours=0,
        metadata_sha256=sha256(tmp_path / "ICOSETC_ZZ-Test_SITEINFO_L2.csv"),
        physical_group_id="physical",
        spatial_group_id="spatial",
        split="train",
        frequency_bin=0,
        lon=9.0,
        lat=45.0,
        landcover=30,
        climate_class=1,
    )
    config = dict(
        input_start="2013-03-18",
        start="2021-12-27",
        end="2022-01-01",
        as_of="2022-02-01",
        canopy_max_age_days=366,
    )
    result = process_station(row, tmp_path / "output", config)
    assert result["footprint_tile_valid"] == 1
    candidate = result["candidates"][0]
    assert candidate["et"] == pytest.approx(100 * DAY / 2450000)
    group = zarr.open_group(candidate["footprint_zarr"], mode="r")
    assert 0.95 < group["mass"][:].sum() < 1
    assert group["weights"][:].sum() == pytest.approx(1, abs=1e-6)
    assert group["interval_seconds"][:].sum() == 5 * DAY
    assert group.attrs["supervision_ready"] is False
    assert process_station(row, tmp_path / "output", config) == result
