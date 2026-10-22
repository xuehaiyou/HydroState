"""Concurrency must not relax quotas, overlap exclusion or location limits."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from hydrostate.data.retrieval.sampling import SamplingLedger
from hydrostate.data.retrieval.scheduling import completed_samples


def sample(index, **kwargs):
    return dict(
        crs="EPSG:32650",
        transform=[10, 0, index * 2000, 0, -10, 0],
        frequency_bin=0,
        target_time="2020-01-01",
        **kwargs,
    )


def test_finishing_worker_refills_without_waiting_for_slow_partner():
    slow_started, third_started = Event(), Event()
    rows = [sample(i) for i in range(3)]

    def collect(row):
        if row is rows[0]:
            slow_started.set()
            assert third_started.wait(5), "Worker waited for the whole batch"
        elif row is rows[1]:
            assert slow_started.wait(5)
        else:
            third_started.set()
        return row

    ledger = SamplingLedger()
    with ThreadPoolExecutor(max_workers=2) as workers:
        result = list(completed_samples(workers, rows, ledger, {0: 3}, 2, collect))
    assert len(result) == 3 and ledger.counts[0] == 3


def test_rejected_reservation_releases_quota_and_overlapping_neighbour():
    rows = [sample(0), sample(0), sample(2), sample(3)]
    called = []

    def collect(row):
        called.append(row)
        return None if row is rows[0] else row

    ledger = SamplingLedger()
    with ThreadPoolExecutor(max_workers=4) as workers:
        result = list(completed_samples(workers, rows, ledger, {0: 1}, 4, collect))
    assert result == [None, rows[1]] and len(called) == 2
    assert ledger.counts[0] == 1


def test_committed_plus_inflight_dates_do_not_exceed_location_limit():
    committed = [dict(sample(0), target_time=d) for d in ("2019-01-01", "2019-03-01", "2019-05-01")]
    rows = [dict(sample(0), target_time=d) for d in ("2020-01-01", "2020-03-01")]
    ledger = SamplingLedger(committed, max_dates=4)
    with ThreadPoolExecutor(max_workers=4) as workers:
        result = list(completed_samples(workers, rows, ledger, {0: 10}, 4, lambda row: row))
    assert len(result) == 1 and ledger.counts[0] == 4


def test_network_failure_propagates_without_consuming_quota():
    def fail(row):
        raise ConnectionError("GEE unavailable")

    ledger = SamplingLedger()
    with ThreadPoolExecutor(max_workers=2) as workers:
        with pytest.raises(ConnectionError, match="GEE unavailable"):
            list(completed_samples(workers, [sample(0)], ledger, {0: 1}, 2, fail))
    assert ledger.counts[0] == 0
