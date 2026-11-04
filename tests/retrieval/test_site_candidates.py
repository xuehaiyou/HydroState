from datetime import date, timedelta

from hydrostate.data.site.candidates import StratifiedPicker, spaced_capacity


def candidates():
    result = []
    for bin_id, sites in [(0, 3), (5, 1)]:
        for site in range(sites):
            for day in range(0, 200, 8):
                stamp = date(2020, 1, 1) + timedelta(days=day)
                result.append(
                    dict(
                        candidate_id=f"{bin_id}_{site}_{day}",
                        frequency_bin=bin_id,
                        physical_group_id=f"{bin_id}_{site}",
                        landcover=10 + site,
                        climate_class=8,
                        year=stamp.year,
                        month=stamp.month,
                        target_time=stamp.isoformat(),
                    )
                )
    return result


def test_sparse_frequency_gets_many_dates_and_success_only_counts():
    picker = StratifiedPicker(candidates(), seed=3)
    first = picker.next({0: 6, 5: 6})
    picker.record(first, success=False)
    chosen = []
    while row := picker.next({0: 6, 5: 6}):
        chosen.append(row)
        picker.record(row, success=True)
    assert picker.counts == {0: 6, 5: 6}
    assert len({r["physical_group_id"] for r in chosen if r["frequency_bin"] == 5}) == 1
    assert first not in chosen
    for dates in picker.dates.values():
        dates = sorted(dates)
        assert all(b - a >= 30 * 86400 for a, b in zip(dates, dates[1:]))


def test_candidate_capacity_dedupes_sensors_and_respects_year_boundary():
    assert spaced_capacity(["2020-12-26", "2021-01-01", "2021-01-25", "2021-01-25"]) == 2


def test_exhausted_bin_does_not_block_other_quota_and_seed_reproducible():
    choices = []
    for _ in range(2):
        picker = StratifiedPicker(candidates(), seed=3)
        selected = []
        while row := picker.next({0: 5, 3: 5}):
            selected.append(row["candidate_id"])
            picker.record(row, success=True)
        assert picker.counts[0] == 5
        assert picker.counts[3] == 0
        choices.append(selected)
    assert choices[0] == choices[1]


def test_bucket_sampler_matches_original_lexicographic_search():
    from hydrostate.data.site.temporal import epoch

    rows = candidates()
    # Include a nearby centre in another water group; the date gap is still shared.
    rows += [dict(r, candidate_id="other_" + r["candidate_id"], frequency_bin=3) for r in rows[:30]]
    picker = StratifiedPicker(rows, seed=17)
    quotas = {0: 30, 3: 30, 5: 30}
    attempts = 0
    while True:
        expected = None
        bins = sorted(
            (b for b in quotas if picker.counts[b] < quotas[b]),
            key=lambda b: (picker.counts[b] / quotas[b], b),
        )
        for b in bins:
            eligible = [
                r
                for r in picker.rows
                if r["frequency_bin"] == b
                and r["candidate_id"] not in picker.attempted
                and all(
                    abs(epoch(r["target_time"]) - d) >= 30 * 86400
                    for d in picker.dates[r["physical_group_id"]]
                )
            ]
            if not eligible:
                continue

            def priority(r):
                space = (b, r["landcover"], r["climate_class"])
                year = (*space, r["year"])
                return (
                    picker.space[space],
                    picker.years[year],
                    picker.months[(*year, r["month"])],
                    picker.sites[(b, r["physical_group_id"])],
                )

            expected = min(eligible, key=priority)
            break
        actual = picker.next(quotas)
        assert actual == expected
        if actual is None:
            break
        picker.record(actual, success=attempts % 7 != 0)
        attempts += 1
    assert attempts > 10


def test_quality_candidate_checkpoint_keeps_real_mornings_and_provenance(tmp_path):
    from hydrostate.data.site.candidates import _read_candidate_file

    path = tmp_path / "N_S_sm_sensor.stm"
    path.write_text(
        "N N S 41 25 600 0.05 0.05 Sensor\n"
        + "".join(f"2021/10/{d:02d} 04:00 0.2 G M\n" for d in range(8, 16))
    )
    (tmp_path / "labels").mkdir()
    row = dict(
        source_path=str(path),
        source_bytes=path.stat().st_size,
        source_mtime_ns=path.stat().st_mtime_ns,
        file_id="test",
        site_id="ismn:N:S",
        physical_group_id="ismn:N:S",
        spatial_group_id="spatial",
        split="train",
        lon=25.0,
        lat=41.0,
        frequency_bin=0,
        landcover=10,
        climate_class=8,
        time_start="2021-10-08T04:00:00",
        time_end="2021-10-15T04:01:00",
    )
    config = dict(input_start="2013-03-18", as_of="2026-11-03")
    summary, rows = _read_candidate_file((row, str(tmp_path), config))
    assert summary["valid_windows"] == 1
    assert rows[0]["valid_days"] == 8
    assert rows[0]["target_time"] == "2021-10-08"
    assert _read_candidate_file((row, str(tmp_path), config)) == (summary, rows)
