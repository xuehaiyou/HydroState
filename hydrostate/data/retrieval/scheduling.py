"""Rolling workers with just-in-time dates and successful + in-flight balance."""

from collections import defaultdict, deque
from concurrent.futures import FIRST_COMPLETED, wait
from datetime import date

from .sampling import ICE_GROUP, SamplingLedger, location_key


class SamplePlanner:
    """Choose a centre and a date only when a worker slot becomes available."""

    def __init__(self, candidates, dates, ledger, rng, tried):
        self.rng, self.tried = rng, tried
        self.queues = defaultdict(deque)
        self.calendar = defaultdict(list)
        for day in sorted(dates):
            when = date.fromisoformat(day)
            self.calendar[when.year, when.month].append(day)
        self.seen = set()
        self.extend(candidates, ledger)

    def extend(self, candidates, ledger):
        candidates = list(candidates)
        self.rng.shuffle(candidates)
        for sample in candidates:
            identity = location_key(sample), sample.get("candidate_target_time")
            if identity not in self.seen:
                self.put(sample, ledger)

    def count(self, groups):
        return sum(len(queue) for key, queue in self.queues.items() if key[0] in groups)

    def put(self, sample, ledger):
        visits = len(ledger.dates.get(location_key(sample), ()))
        if visits < ledger.max_dates:
            self.seen.add((location_key(sample), sample.get("candidate_target_time")))
            self.queues[
                sample["sampling_group"],
                visits,
                sample.get("climate_class", 0),
                sample.get("candidate_target_time"),
                sample.get("split", ""),
                tuple(sample.get("block", ())),
            ].append(sample)

    def choose_date(self, sample, ledger, reserved):
        key = location_key(sample)
        if len(ledger.dates.get(key, ())) + len(reserved.dates.get(key, ())) >= ledger.max_dates:
            return None
        if fixed := sample.get("candidate_target_time"):
            period = int(fixed[:4]), int(fixed[5:7])
            if fixed not in self.calendar.get(period, ()):
                raise ValueError("Ice candidate date is outside the product calendar")
            return (
                fixed
                if (
                    (key, fixed) not in self.tried
                    and ledger.allows(sample, fixed)
                    and reserved.allows(sample, fixed)
                )
                else None
            )
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
                chosen = min(
                    available,
                    key=lambda k: (
                        (ledger.counts[k[0]] + reserved.counts[k[0]]) / target[k[0]],
                        k[1],
                        ledger.climates[k[2]]
                        + reserved.climates[k[2]]
                        + (
                            ledger.years[int(k[3][:4])]
                            + reserved.years[int(k[3][:4])]
                            + ledger.months[int(k[3][5:7])]
                            + reserved.months[int(k[3][5:7])]
                            if k[3]
                            else 0
                        ),
                        (ledger.splits[k[4]] + reserved.splits[k[4]])
                        / {"train": 0.7, "validation": 0.15, "test": 0.15}.get(k[4], 1),
                        ledger.blocks[k[5]] + reserved.blocks[k[5]],
                    ),
                )
                sample = self.queues[chosen].popleft()
                self.seen.discard((location_key(sample), sample.get("candidate_target_time")))
                day = self.choose_date(sample, ledger, reserved)
                if day is None:
                    # A pending neighbour may temporarily block all dates.
                    # Revisit after completion, without spinning in this call.
                    empty = SamplingLedger(max_dates=ledger.max_dates, min_days=ledger.min_days)
                    if self.choose_date(sample, ledger, empty) is not None:
                        deferred.append(sample)
                    continue
                self.tried.add((location_key(sample), day))
                return dict(sample, target_time=day)
        finally:
            for sample in deferred:
                self.put(sample, ledger)


def completed_samples(
    workers,
    candidates,
    ledger,
    target,
    concurrency,
    collect_one,
    *,
    dates,
    rng,
    tried,
    ice_source=None,
    next_candidates=None,
    ice_buffer=16,
):
    """Sample downloads and one DW query share the same worker budget.

    Water-pool refills happen only after active workers drain: their metadata
    fetcher can then use all four slots without overlapping another four jobs.
    """
    planner, pending = SamplePlanner(candidates, dates, ledger, rng, tried), {}
    water_exhausted = next_candidates is None
    try:
        while True:
            water_groups = {k for k in target if k != ICE_GROUP and ledger.counts[k] < target[k]}
            refill = bool(water_groups) and not planner.count(water_groups) and not water_exhausted
            if refill and not pending:
                batch = next_candidates()
                if batch is None:
                    water_exhausted = True
                else:
                    planner.extend(batch, ledger)
                continue
            reserved = SamplingLedger(
                [item for kind, item in pending.values() if kind == "sample"],
                ledger.max_dates,
                ledger.min_days,
            )
            ice_needed = (
                ICE_GROUP in target
                and ledger.counts[ICE_GROUP] + reserved.counts[ICE_GROUP] < target[ICE_GROUP]
            )
            if (
                not refill
                and ice_source is not None
                and ice_needed
                and planner.count({ICE_GROUP})
                < min(
                    ice_buffer,
                    target[ICE_GROUP] - ledger.counts[ICE_GROUP] - reserved.counts[ICE_GROUP],
                )
                and len(pending) < concurrency
                and not any(kind == "query" for kind, _ in pending.values())
            ):
                query = ice_source.next_query(ledger, reserved)
                if query is not None:
                    pending[workers.submit(ice_source.fetch, query)] = ("query", query)
            while not refill and len(pending) < concurrency:
                sample = planner.next(ledger, reserved, target)
                if sample is None:
                    break
                pending[workers.submit(collect_one, sample)] = ("sample", sample)
                reserved.add(sample)
            if not pending:
                if not water_exhausted and water_groups:
                    continue
                return
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                kind, _ = pending.pop(future)
                result = future.result()
                if kind == "query":
                    planner.extend(result, ledger)
                    continue
                if result is not None:
                    ledger.add(result)
                    if ice_source is not None:
                        ice_source.observe(result)
                yield result
    finally:
        for future in pending:
            future.cancel()
