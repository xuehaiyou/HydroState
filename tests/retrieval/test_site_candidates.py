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
