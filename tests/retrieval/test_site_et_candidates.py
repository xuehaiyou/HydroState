import json

import pandas as pd
import pytest

from hydrostate.data.site.archives import sha256
from hydrostate.data.site.et_candidates import prepare_et_candidates
from hydrostate.data.site.sampling_capacity import bundle_candidates, draw_capacity, sample_capacity
from hydrostate.data.site.temporal import DAY


def make_indices(tmp_path):
    index, splits, strata = [tmp_path / name for name in ("index", "splits", "strata")]
    for path in (index, splits, strata):
        path.mkdir()
    files, locations, properties = [], [], []
    # A missing first day still leaves 7/8 valid days. No footprint variables exist.
    times = pd.date_range("2021-01-02", "2021-01-09", freq="h", inclusive="left")
    for i, source in enumerate(["fluxnet", "icos"]):
        path = tmp_path / f"{source}_HH.csv"
        pd.DataFrame(
            dict(
                TIMESTAMP_START=times.strftime("%Y%m%d%H%M"),
                TIMESTAMP_END=(times + pd.Timedelta(hours=1)).strftime("%Y%m%d%H%M"),
                LE_F_MDS=100.0,
                LE_F_MDS_QC=0,
            )
        ).to_csv(path, index=False)
        metadata = tmp_path / f"{source}_metadata.csv"
        metadata.write_text("indexed verified metadata\n")
        files.append(
            dict(
                file_id=f"file_{i}",
                source=source,
                site_id=f"{source}:X",
                source_path=str(path),
                source_bytes=path.stat().st_size,
                source_mtime_ns=path.stat().st_mtime_ns,
                metadata_file=str(metadata),
                metadata_sha256=sha256(metadata),
                utc_offset_hours=0.0,
                lon=9.0,
                lat=45.0,
                source_landcover="GRA",
                time_start="2021-01-02T00:00:00",
                time_end="2021-01-09T00:00:00",
                metadata_issue="",
                columns=json.dumps(list(pd.read_csv(path, nrows=0).columns)),
            )
        )
        locations.append(
            dict(
                site_id=f"{source}:X",
                lon=9.0,
                lat=45.0,
                location_id=f"location_{i}",
                physical_group_id="same_station",
                spatial_group_id="space",
                split="train",
            )
        )
        properties.append(
            dict(location_id=f"location_{i}", frequency_bin=3, landcover=30, climate_class=8)
        )
    pd.DataFrame(files).to_parquet(index / "source_files.parquet", index=False)
    pd.DataFrame(locations).to_parquet(splits / "station_splits.parquet", index=False)
    pd.DataFrame(properties).to_parquet(strata / "station_strata.parquet", index=False)
    (strata / "strata_report.json").write_text(json.dumps(dict(input_start="2013-03-18")))
    return index, splits, strata


def test_full_temporal_qc_includes_no_footprint_and_reuses_partial_run(tmp_path):
    index, splits, strata = make_indices(tmp_path)
    output = tmp_path / "quality"
    partial = prepare_et_candidates(
        index, splits, strata, output, as_of="2021-02-01", limit=1, workers=1
    )
    assert not partial["complete"] and partial["valid_source_windows"] == 1
    with pytest.raises(ValueError, match="complete"):
        sample_capacity(
            tmp_path / "missing_sm",
            output / "candidates.parquet",
            tmp_path / "missing_footprint",
            tmp_path / "selection",
        )
    report = prepare_et_candidates(
        index, splits, strata, output, as_of="2021-02-01", workers=1, limit=9
    )
    assert report["complete"] and report["valid_source_windows"] == 2
    assert sum(r["reused_checkpoint"] for r in report["files"]) == 1
    table = pd.read_parquet(output / "candidates.parquet")
    assert set(table.target_time) == {"2021-01-01"}
    assert set(table.source) == {"fluxnet", "icos"}
    assert table.et.tolist() == pytest.approx([100 * DAY / 2450000] * 2)
    assert table.valid_fraction.tolist() == pytest.approx([7 / 8] * 2)
    assert not table.spatial_support_verified.any()
    assert not table.satellite_inputs_checked.any()
    assert report["candidates_sha256"] == sha256(output / "candidates.parquet")
    members = table.assign(target="et", operator_ready=False)
    bundled, references = bundle_candidates([members])
    assert len(bundled) == 1 and len(references) == 2
    assert bundled.iloc[0].source == "fluxnet"
    selected, plan = draw_capacity(bundled)
    assert len(selected) == 1 and plan["selected"][3] == 1


def test_bundle_keeps_distinct_centres_and_gap_crosses_sources_targets(tmp_path):
    index, splits, strata = make_indices(tmp_path)
    output = tmp_path / "quality"
    prepare_et_candidates(index, splits, strata, output, as_of="2021-02-01", workers=1)
    et = (
        pd.read_parquet(output / "candidates.parquet")
        .iloc[:1]
        .assign(target="et", operator_ready=False)
    )
    sm = et.assign(candidate_id="sm", target="sm", source="ismn", operator_ready=True)
    near = sm.assign(
        candidate_id="near",
        lon=9.0001,
        frequency_bin=3,
        target_time="2021-01-09",
        target_end="2021-01-17",
    )
    bundled, members = bundle_candidates([et, sm, near])
    assert len(bundled) == 2 and len(members) == 3
    both = bundled[bundled.lon.eq(9.0)].iloc[0]
    assert both.has_sm and both.has_et and not both.has_et_footprint
    chosen, _ = draw_capacity(bundled)
    assert len(chosen) == 1  # The same physical station cannot evade the gap via another centre.
    with pytest.raises(ValueError, match="split"):
        bundle_candidates([et, sm.assign(split="test")])


def test_date_choice_never_sacrifices_frequency_capacity_for_other_strata():
    table = pd.DataFrame(
        [
            dict(
                candidate_id=str(i),
                physical_group_id="one",
                frequency_bin=5,
                split="train",
                landcover=10 + i,
                climate_class=8,
                year=2021,
                month=1 if i < 2 else 2,
                target_time=day,
            )
            for i, day in enumerate(["2021-01-01", "2021-01-25", "2021-02-02"])
        ]
    )
    for seed in range(5):
        chosen, report = draw_capacity(table, seed=seed)
        assert set(chosen.target_time) == {"2021-01-01", "2021-02-02"}
        assert report["selected"][5] == report["quotas"][5] == 2


def test_capacity_draw_is_seeded_and_reaches_all_station_maxima():
    import random
    from datetime import date, timedelta

    from hydrostate.data.site.candidates import spaced_capacity

    rng = random.Random(77)
    rows = []
    for site in range(5):
        for day in sorted(rng.sample(range(800), 80)):
            stamp = date(2020, 1, 1) + timedelta(days=day)
            rows.append(
                dict(
                    candidate_id=f"{site}_{day}",
                    physical_group_id=f"site_{site}",
                    frequency_bin=site % 3,
                    split="train",
                    landcover=10 + site,
                    climate_class=8,
                    year=stamp.year,
                    month=stamp.month,
                    target_time=stamp.isoformat(),
                )
            )
    table = pd.DataFrame(rows)
    first, _ = draw_capacity(table, seed=8)
    second, _ = draw_capacity(table, seed=8)
    other, _ = draw_capacity(table, seed=9)
    assert first.candidate_id.tolist() == second.candidate_id.tolist()
    assert first.candidate_id.tolist() != other.candidate_id.tolist()
    assert len(first) == sum(
        spaced_capacity(g.target_time) for _, g in table.groupby("physical_group_id")
    )
