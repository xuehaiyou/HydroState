import csv

import pytest

from hydrostate.data.site.index import bif_metadata, file_edges, source_record


def test_index_preserves_deep_station_but_does_not_enable_sm(tmp_path):
    path = tmp_path / "N_S_sm_0.1_0.1.stm"
    path.write_text(
        "N N S 41 25 600 0.1 0.1 Sensor\n2014/01/01 04:00 0.2 G M\n2015/12/31 04:00 0.3 G M\n"
    )
    row = source_record(path, "ismn", {})
    assert row["lat"] == 41
    assert not row["sm_header_eligible"]
    assert row["time_start"] == "2014-01-01T04:00:00"
    assert row["time_end"] == "2015-12-31T04:01:00"
    assert "Depth" in row["eligibility_reason"]


def test_flux_edges_comments_and_no_qc_claim(tmp_path):
    path = tmp_path / "table.csv"
    path.write_text(
        "# Site: US-X\n# Version: 1\nTIMESTAMP_START,TIMESTAMP_END,LE\n"
        "201501010000,201501010030,-9999\n201501010030,201501010100,4\n"
    )
    columns, start, end = file_edges(path)
    assert columns[-1] == "LE"
    assert (start, end) == ("2015-01-01T00:00:00", "2015-01-01T01:00:00")


def test_bif_multiple_locations_do_not_pick_last(tmp_path):
    path = tmp_path / "site.csv"
    with path.open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["SITE_ID", "VARIABLE", "DATAVALUE"])
        for name, value in [
            ("LOCATION_LAT", 41),
            ("LOCATION_LAT", 42),
            ("LOCATION_LONG", 25),
            ("UTC_OFFSET", 0),
        ]:
            writer.writerow(["XX-X", name, value])
    row = bif_metadata(path, "ameriflux", "XX-X")
    assert row["lat"] is None
    assert row["lon"] == 25
    assert "LOCATION_LAT" in row["metadata_issue"]
    assert row["group_id"] == "flux:XX-X"


def test_bif_xlsx_uses_same_metadata_convention(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    book = openpyxl.Workbook()
    sheet = book.active
    sheet.append(["SITE_ID", "VARIABLE", "DATAVALUE"])
    for name, value in [("LOCATION_LAT", 41), ("LOCATION_LONG", 25), ("UTC_OFFSET", 0)]:
        sheet.append(["XX-X", name, value])
    path = tmp_path / "site.xlsx"
    book.save(path)
    row = bif_metadata(path, "ameriflux", "XX-X")
    assert row["metadata_issue"] == ""
