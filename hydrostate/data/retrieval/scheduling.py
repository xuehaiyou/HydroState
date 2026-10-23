"""Rolling workers with just-in-time dates and successful + in-flight balance."""

from collections import defaultdict, deque
from concurrent.futures import FIRST_COMPLETED, wait
from datetime import date

from .sampling import SamplingLedger, location_key


class SamplePlanner:
    """Choose a centre and a date only when a worker slot becomes available."""

    def __init__(self, candidates, dates, ledger, rng, tried):
        self.rng, self.tried = rng, tried
        self.queues = defaultdict(deque)
        self.calendar = defaultdict(list)
        for day in sorted(dates):
            when = date.fromisoformat(day)
            self.calendar[when.year, when.month].append(day)
        candidates = list(candidates)
        rng.shuffle(candidates)
        for sample in candidates:
            self.put(sample, ledger)

    def put(self, sample, ledger):
        visits = len(ledger.dates.get(location_key(sample), ()))
        if visits < ledger.max_dates:
            self.queues[sample["frequency_bin"], visits, sample.get("climate_class", 0)].append(
                sample
            )

    def choose_date(self, sample, ledger, reserved):
        key = location_key(sample)
        if len(ledger.dates.get(key, ())) + len(reserved.dates.get(key, ())) >= ledger.max_dates:
            return None
        # Shuffle before stable sorting: equal priorities must not all select
        # the same date. In-flight counts change immediately after each submit.
        periods = list(self.calendar)
        self.rng.shuffle(periods)
        periods.sort(
            key=lambda p: (
                ledger.years[p[0]]
                + reserved.years[p[0]]
                + ledger.months[p[1]]
                + reserved.months[p[1]]
            )
        )
        for period in periods:
            days = list(self.calendar[period])
            self.rng.shuffle(days)
            for day in days:
                if (
                    (key, day) not in self.tried
                    and ledger.allows(sample, day)
                    and reserved.allows(sample, day)
                ):
                    return day
        return None

    def next(self, ledger, reserved, target):
        deferred = []
        try:
            while True:
                available = [
                    k
                    for k, queue in self.queues.items()
                    if queue and ledger.counts[k[0]] + reserved.counts[k[0]] < target[k[0]]
                ]
                if not available:
                    return None
                self.rng.shuffle(available)
                fb, visits, climate = min(
                    available,
                    key=lambda k: (
                        (ledger.counts[k[0]] + reserved.counts[k[0]]) / target[k[0]],
                        k[1],
                        ledger.climates[k[2]] + reserved.climates[k[2]],
                    ),
                )
                sample = self.queues[fb, visits, climate].popleft()
                day = self.choose_date(sample, ledger, reserved)
                if day is None:
                    # A pending neighbour may temporarily block all dates.
                    # Revisit after completion, without spinning in this call.
                    deferred.append(sample)
                    continue
                self.tried.add((location_key(sample), day))
                return dict(sample, target_time=day)
        finally:
            for sample in deferred:
                self.put(sample, ledger)


def completed_samples(
    workers, candidates, ledger, target, concurrency, collect_one, *, dates, rng, tried
):
    """Keep workers busy; a rejection releases every reservation, not just quota.

    Only this thread owns the ledgers. Each centre is attempted once per pool;
    subsequent pools supply fresh centres. Later cached passes may try new dates.
    Network errors propagate instead of being recorded as unsuitable geography.
    """
    planner, pending = SamplePlanner(candidates, dates, ledger, rng, tried), {}
    try:
        while True:
            reserved = SamplingLedger(pending.values(), ledger.max_dates, ledger.min_days)
            while len(pending) < concurrency:
                sample = planner.next(ledger, reserved, target)
                if sample is None:
                    break
                pending[workers.submit(collect_one, sample)] = sample
                reserved.add(sample)
            if not pending:
                return
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                pending.pop(future)
                row = future.result()
                if row is not None:
                    ledger.add(row)
                yield row
    finally:
        for future in pending:
            future.cancel()
