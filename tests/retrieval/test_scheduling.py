"""Rolling sampling balances before submission and preserves hard constraints."""

import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event, Lock

import pytest

from hydrostate.data.retrieval.candidates import WaterCandidateSource
from hydrostate.data.retrieval.sampling import SamplingLedger, location_key, quotas
from hydrostate.data.retrieval.scheduling import SamplePlanner, completed_samples


def sample(index, **kwargs):
    return (
        dict(
            crs="EPSG:32650",
            transform=[10, 0, index * 2000, 0, -10, 0],
            frequency_bin=0,
            sampling_group=0,
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
                capacity=max(128, len(rows)),
                ice_buffer=max(16, len(rows)),
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


def test_fixed_pool_capacity_is_independent_of_total_quota():
    rows = [sample(i, sampling_group=i % 2) for i in range(1000)]
    planner = SamplePlanner([], ["2020-01-01"], SamplingLedger(), random.Random(0), set())
    planner.extend(rows, SamplingLedger(), limits={0: 128, 1: 128})
    assert planner.count({0}) == planner.count({1}) == 128
    planner.extend(rows, SamplingLedger(), limits={0: 128, 1: 128})
    assert len(planner.snapshot()) == 256


def test_water_cursor_resumes_and_exhausts_without_an_inner_executor(monkeypatch):
    import hydrostate.data.retrieval.candidates as candidates

    monkeypatch.setattr(candidates, "global_blocks", lambda cfg: [(0, 0), (2, 0)])

    class Catalog:
        def __init__(self, *args):
            pass

        def fetch(self, query):
            return [sample(query[0])]

    monkeypatch.setattr(candidates, "CandidateCatalog", Catalog)
    cfg = dict(max_dates_per_location=2)
    first = WaterCandidateSource(None, cfg)
    assert first.next_query(None, None) == 0
    second = WaterCandidateSource(None, cfg, first.snapshot())
    assert [second.next_query(None, None) for _ in range(4)] == [1, 0, 1, None]
    assert len(second.fetch(0)) == 1


def test_full_pilot_simulation_keeps_quotas_dates_and_fresh_locations_after_failures():
    target, ledger = quotas(160), SamplingLedger()
    dates = [f"{year}-{month:02d}-01" for year in range(2017, 2025) for month in range(1, 13)]
    rows = [
        sample(i, frequency_bin=i % 7, sampling_group=i % 8, climate_class=i % 5)
        for i in range(480)
    ]
    accepted = [
        s
        for s in run(
            rows, ledger, target, lambda row: None if row["index"] % 11 == 0 else row, dates=dates
        )
        if s is not None
    ]
    assert len(accepted) == len(ledger.dates) == 160
    assert ledger.counts == Counter(target)
    assert len(ledger.years) == 8 and len(ledger.months) == 12
    assert max(ledger.years.values()) <= 26 and max(ledger.months.values()) <= 20
    assert len(ledger.climates) == 5


def test_low_water_refill_starts_before_empty_and_queue_size_is_fixed():
    ledger, observed = SamplingLedger(), []

    class Source:
        index = 0

        def next_query(self, *args):
            if self.index >= 4:
                return None
            self.index += 1
            return self.index

        def fetch(self, query):
            return [sample(query * 100 + i) for i in range(30)]

        def snapshot(self):
            return dict(index=self.index)

    source = Source()
    with ThreadPoolExecutor(max_workers=4) as workers:
        results = list(
            completed_samples(
                workers,
                [sample(i) for i in range(3)],
                ledger,
                {0: 15},
                4,
                lambda r: r,
                dates=["2020-01-01"],
                rng=random.Random(0),
                tried=set(),
                water_source=source,
                capacity=8,
                low_watermark=3,
                checkpoint=lambda state: observed.append(state),
            )
        )
    assert len(results) == 15
    assert observed[0]["pending_queries"], "Low-water refill was not started"
    assert observed[0]["pending_samples"], "Existing candidates waited for the refill"
    assert all(len(s["candidates"]) <= 8 for s in observed)


def test_stop_resume_and_extension_retain_unused_candidates_and_global_ledger():
    stop, saved, ledger = Event(), [], SamplingLedger()
    rows = [sample(i) for i in range(12)]

    def collect(row):
        if row["index"] % 2 == 0:
            stop.set()
        return row

    with ThreadPoolExecutor(max_workers=2) as workers:
        list(
            completed_samples(
                workers,
                rows,
                ledger,
                {0: 8},
                2,
                collect,
                dates=["2020-01-01"],
                rng=random.Random(1),
                tried=set(),
                stop=stop,
                capacity=12,
                low_watermark=3,
                checkpoint=lambda s: saved.append(s),
            )
        )
    before = ledger.counts[0]
    assert 0 < before < 8 and saved[-1]["candidates"]
    with ThreadPoolExecutor(max_workers=2) as workers:
        list(
            completed_samples(
                workers,
                [],
                ledger,
                {0: 8},
                2,
                lambda r: r,
                dates=["2020-01-01"],
                rng=random.Random(2),
                tried=set(),
                state=saved[-1],
                capacity=12,
                low_watermark=3,
                checkpoint=lambda s: saved.append(s),
            )
        )
    assert ledger.counts[0] == 8 and len(saved[-1]["candidates"]) == 4
    with ThreadPoolExecutor(max_workers=2) as workers:
        list(
            completed_samples(
                workers,
                [],
                ledger,
                {0: 12},
                2,
                lambda r: r,
                dates=["2020-01-01"],
                rng=random.Random(3),
                tried=set(),
                state=saved[-1],
                capacity=12,
            )
        )
    assert ledger.counts[0] == len(ledger.dates) == 12


def test_failed_query_is_replayed_after_source_cursor_has_advanced():
    saved, ledger = [], SamplingLedger()

    class Source:
        def __init__(self, fail):
            self.fail, self.cursor = fail, 0

        def next_query(self, *args):
            self.cursor += 1
            return 4 if self.cursor == 1 else None

        def fetch(self, query):
            assert query == 4
            if self.fail:
                raise ConnectionError("network")
            return [sample(4)]

        def snapshot(self):
            return dict(cursor=self.cursor)

    with ThreadPoolExecutor(max_workers=1) as workers:
        with pytest.raises(ConnectionError):
            list(
                completed_samples(
                    workers,
                    [],
                    ledger,
                    {0: 1},
                    1,
                    lambda r: r,
                    dates=["2020-01-01"],
                    rng=random.Random(0),
                    tried=set(),
                    water_source=Source(True),
                    checkpoint=lambda s: saved.append(s),
                )
            )
    resumed = Source(False)
    resumed.cursor = 1
    with ThreadPoolExecutor(max_workers=1) as workers:
        result = list(
            completed_samples(
                workers,
                [],
                ledger,
                {0: 1},
                1,
                lambda r: r,
                dates=["2020-01-01"],
                rng=random.Random(0),
                tried=set(),
                water_source=resumed,
                state=saved[-1],
            )
        )
    assert len(result) == ledger.counts[0] == 1


def test_blocked_full_buffer_refills_instead_of_reporting_exhaustion():
    committed = [dict(sample(i), target_time="2020-01-01") for i in range(5)]
    ledger = SamplingLedger(committed)

    class Source:
        called = 0

        def next_query(self, *args):
            self.called += 1
            return 0 if self.called == 1 else None

        def fetch(self, query):
            return [sample(100)]

    source = Source()
    with ThreadPoolExecutor(max_workers=1) as workers:
        results = list(
            completed_samples(
                workers,
                [sample(i) for i in range(5)],
                ledger,
                {0: 6},
                1,
                lambda r: r,
                dates=["2020-01-01"],
                rng=random.Random(0),
                tried=set(),
                capacity=5,
                low_watermark=1,
                water_source=source,
            )
        )
    assert len(results) == 1 and results[0]["index"] == 100
    assert ledger.counts[0] == 6
