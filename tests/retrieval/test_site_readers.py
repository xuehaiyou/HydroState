import csv
import json

import pytest

from hydrostate.data.site.prepare import prepare_labels
from hydrostate.data.site.readers import (
    UnsupportedSource,
    fluxnet_metadata,
    ismn_header,
    ismn_windows,
)


def ismn_file(tmp_path, depth="0.05 0.05", sensor="Sentek-DaD"):
    path = tmp_path / "network_station_sm_0.050000_sensor.stm"
    path.write_text(
        f"NETWORK NETWORK Station 41 25 600 {depth} '{sensor}'\n"
        + "".join(f"2021/10/{day:02d} 04:00 {day / 100} G M\n" for day in range(8, 16))
    )
    return path


def test_ismn_raw_windows_and_no_daily_reconstruction(tmp_path):
    path = ismn_file(tmp_path)
    rows = ismn_windows(path, "2021-10-08", "2021-10-16", "2021-11-30")
    assert len(rows) == 1
    assert rows[0]["sm"]["value"] == pytest.approx(0.115)
    assert rows[0]["sm"]["valid_days"] == 8
    assert rows[0]["depth_from_m"] == 0.05
    assert rows[0]["support_type"] == "point"
    assert len(rows[0]["source_file"]["sha256"]) == 64
    out = tmp_path / "labels"
    prepare_labels([path], out, "ismn", "2021-10-08", "2021-10-16", "2021-11-30")
    prepare_labels([path], out, "ismn", "2021-10-08", "2021-10-16", "2021-11-30")
    assert json.loads((out / "report.json").read_text())["files"][0]["sm"] == 1
    with pytest.raises(ValueError, match="configuration changed"):
        prepare_labels([path], out, "ismn", "2021-10-08", "2021-10-16", "2022-11-30")


def test_depth_and_nonpoint_support_not_guessed(tmp_path):
    with pytest.raises(UnsupportedSource, match="Depth"):
        ismn_header(ismn_file(tmp_path, "0.1 0.1"))
    with pytest.raises(UnsupportedSource, match="Non-point"):
        ismn_header(ismn_file(tmp_path, sensor="Hydroinnova"))


def test_historical_offset_requires_explicit_resolution(tmp_path):
    path = tmp_path / "AMF_US-X_BIF_2020.csv"
    fields = ["SITE_ID", "VARIABLE", "DATAVALUE"]
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for key, value in [
            ("LOCATION_LAT", 40),
            ("LOCATION_LONG", -80),
            ("UTC_OFFSET", -5),
            ("UTC_OFFSET", -6),
        ]:
            writer.writerow(dict(SITE_ID="US-X", VARIABLE=key, DATAVALUE=value))
    with pytest.raises(UnsupportedSource, match="historical UTC_OFFSET"):
        fluxnet_metadata(tmp_path / "AMF_US-X_FLUXNET_FLUXMET_HH_2020.csv")


def test_aggregated_sensor_is_not_a_point_even_when_depth_is_shallow(tmp_path):
    path=ismn_file(tmp_path,sensor='Stevens-Hydraprobe-A1-area-weighted')
    with pytest.raises(UnsupportedSource,match='Aggregated sensor support'):
        ismn_header(path)
    assert ismn_header(path,require_shallow=False)['support_type']=='nonpoint_unresolved'
