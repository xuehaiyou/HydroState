"""Rolling sampling balances before submission and preserves hard constraints."""

import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event, Lock

import pytest

from hydrostate.data.retrieval.candidates import candidate_pools
from hydrostate.data.retrieval.sampling import SamplingLedger, location_key, quotas
from hydrostate.data.retrieval.scheduling import SamplePlanner, completed_samples


def sample(index, **kwargs):
    return (
        dict(
            crs="EPSG:32650",
            transform=[10, 0, index * 2000, 0, -10, 0],
            frequency_bin=0,
            climate_class=0,
            index=index,
        )
        | kwargs
    )


def run(rows, ledger, target, collect=lambda row: row, concurrency=4, dates=None, tried=None):
    with ThreadPoolExecutor(max_workers=concurrency) as workers:
        return list(
            completed_samples(
                workers,
                rows,
                ledger,
                target,
                concurrency,
                collect,
                dates=dates or ["2020-01-01"],
                rng=random.Random(43),
                tried=set() if tried is None else tried,
            )
        )


def test_finishing_worker_refills_without_waiting_for_slow_partner():
    slow_started, third_started, lock = Event(), Event(), Lock()
    calls = []

    def collect(row):
        with lock:
            calls.append(row)
            order = len(calls)
        if order == 1:
            slow_started.set()
            assert third_started.wait(5), "Worker waited for the whole batch"
        elif order == 2:
            assert slow_started.wait(5)
        else:
            third_started.set()
        return row

    ledger = SamplingLedger()
    result = run([sample(i) for i in range(3)], ledger, {0: 3}, collect, concurrency=2)
    assert len(result) == 3 and ledger.counts[0] == 3


def test_rejected_reservation_releases_quota_and_overlapping_neighbour():
    rows = [sample(0) for _ in range(3)]
    called = []

    def collect(row):
        called.append(row)
        return None if len(called) == 1 else row

    # Exact duplicate attempts are forbidden even after rejection. Neighbouring
    # centres can still be tried after the failed reservation is released.
    for i, row in enumerate(rows):
        row["transform"][2] += i * 100
    ledger = SamplingLedger()
    result = run(rows, ledger, {0: 1}, collect)
    assert result[0] is None and result[1] is not None and len(called) == 2
    assert ledger.counts[0] == 1


def test_inflight_year_month_and_climate_are_reserved_before_completion():
    barrier, lock, observed = Barrier(4), Lock(), []

    def collect(row):
        with lock:
            observed.append(row)
        barrier.wait(timeout=5)
        return row

    dates = [f"{y}-{m:02d}-01" for y in range(2017, 2021) for m in range(1, 5)]
    run(
        [sample(i, climate_class=i) for i in range(8)],
        SamplingLedger(),
        {0: 4},
        collect,
        dates=dates,
    )
    assert len({s["target_time"][:4] for s in observed}) == 4
    assert len({s["target_time"][5:7] for s in observed}) == 4
    assert len({s["climate_class"] for s in observed}) == 4


def test_committed_plus_inflight_dates_do_not_exceed_location_limit():
    committed = [dict(sample(0), target_time=d) for d in ("2019-01-01", "2019-03-01", "2019-05-01")]
    ledger = SamplingLedger(committed, max_dates=4)
    result = run([sample(0), sample(0)], ledger, {0: 10}, dates=["2020-01-01", "2020-03-01"])
    assert len(result) == 1 and ledger.counts[0] == 4


def test_calendar_uses_remaining_dates_and_resume_rejections():
    row = sample(0)
    ledger = SamplingLedger([dict(row, target_time="2020-01-01")])
    tried = {(location_key(row), "2020-03-01")}
    result = run(
        [row], ledger, {0: 2}, dates=["2020-01-09", "2020-03-01", "2020-05-01"], tried=tried
    )
    assert result[0]["target_time"] == "2020-05-01"
    assert (location_key(row), "2020-05-01") in tried


def test_rejection_releases_soft_counts_too():
    ledger = SamplingLedger([dict(sample(100), target_time="2019-01-01", climate_class=2)])
    planner = SamplePlanner(
        [sample(i, climate_class=i % 2) for i in range(4)],
        ["2019-01-01", "2020-03-01"],
        ledger,
        random.Random(0),
        set(),
    )
    empty = SamplingLedger()
    first = planner.next(ledger, empty, {0: 3})
    assert first["target_time"] == "2020-03-01"
    # Rejection: do not add the row to the committed ledger. The underfilled
    # period remains preferred; neither year nor month is permanently reserved.
    second = planner.next(ledger, empty, {0: 3})
    assert second["target_time"] == "2020-03-01"


def test_network_failure_propagates_without_consuming_quota():
    def fail(row):
        raise ConnectionError("GEE unavailable")

    ledger = SamplingLedger()
    with pytest.raises(ConnectionError, match="GEE unavailable"):
        run([sample(0)], ledger, {0: 1}, fail)
    assert ledger.counts[0] == 0


def test_pool_is_bounded_deduplicated_and_refills_only_unfinished_groups():
    ledger, target = SamplingLedger(), {0: 2, 1: 1}
    batches = [[sample(i, frequency_bin=i % 2) for i in range(j, j + 20)] for j in (0, 20, 40)]
    batches[0] += [sample(0)]
    pools = candidate_pools(batches, ledger, target, random.Random(43), factor=3)
    first = next(pools)
    assert Counter(s["frequency_bin"] for s in first) == {0: 6, 1: 3}
    assert len({location_key(s) for s in first}) == len(first)
    for row in first:
        if row["frequency_bin"] == 1:
            ledger.add(dict(row, target_time="2020-01-01"))
            break
    second = next(pools)
    assert len(second) == 6 and all(s["frequency_bin"] == 0 for s in second)
    assert not {location_key(s) for s in first} & {location_key(s) for s in second}


def test_sparse_pool_yields_available_centres_without_impossible_wait():
    pools = list(
        candidate_pools([[sample(1), sample(2)]], SamplingLedger(), {0: 1, 1: 1}, random.Random(43))
    )
    assert len(pools) == 1 and len(pools[0]) == 2


def test_full_pilot_simulation_keeps_quotas_dates_and_fresh_locations_after_failures():
    target, ledger = quotas(140), SamplingLedger()
    dates = [f"{year}-{month:02d}-01" for year in range(2017, 2025) for month in range(1, 13)]
    rows = [sample(i, frequency_bin=i % 7, climate_class=i % 5) for i in range(420)]
    accepted = [
        s
        for s in run(
            rows, ledger, target, lambda row: None if row["index"] % 4 == 0 else row, dates=dates
        )
        if s is not None
    ]
    assert len(accepted) == len(ledger.dates) == 140
    assert ledger.counts == Counter(target)
    assert len(ledger.years) == 8 and len(ledger.months) == 12
    assert max(ledger.years.values()) <= 24 and max(ledger.months.values()) <= 18
    assert len(ledger.climates) == 5
