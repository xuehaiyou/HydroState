"""Rolling workers with just-in-time dates and successful + in-flight balance."""

import logging
import time
from collections import defaultdict, deque
from concurrent.futures import FIRST_COMPLETED, wait
from datetime import date

from .sampling import ICE_GROUP, SamplingLedger, location_key

LOG = logging.getLogger(__name__)


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

    def extend(self, candidates, ledger, limits=None):
        candidates = list(candidates)
        self.rng.shuffle(candidates)
        for sample in candidates:
            identity = self.identity(sample)
            if identity not in self.seen and (
                limits is None
                or self.count({sample["sampling_group"]}) < limits[sample["sampling_group"]]
            ):
                self.put(sample, ledger)

    @staticmethod
    def identity(sample):
        return location_key(sample), sample.get("candidate_target_time") or sample.get(
            "target_time"
        )

    def snapshot(self):
        return [sample for queue in self.queues.values() for sample in queue]

    def trim(self, limits):
        for group, limit in limits.items():
            excess = self.count({group}) - limit
            for key in list(self.queues):
                if key[0] != group:
                    continue
                while excess > 0 and self.queues[key]:
                    self.seen.discard(self.identity(self.queues[key].pop()))
                    excess -= 1
                if not self.queues[key]:
                    del self.queues[key]

    def count(self, groups):
        return sum(len(queue) for key, queue in self.queues.items() if key[0] in groups)

    def put(self, sample, ledger):
        visits = len(ledger.dates.get(location_key(sample), ()))
        if visits < ledger.max_dates:
            self.seen.add(self.identity(sample))
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
        if fixed := sample.get("candidate_target_time") or sample.get("target_time"):
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
                self.seen.discard(self.identity(sample))
                if not self.queues[chosen]:
                    del self.queues[chosen]
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
    water_source=None,
    capacity=128,
    low_watermark=32,
    ice_buffer=16,
    ice_low_watermark=4,
    state=None,
    checkpoint=None,
    stop=None,
):
    """Queries and downloads share one executor; available groups never wait for others.

    One query may refill a low pool while samples occupy the other slots. When
    there are no ready samples, idle slots can search more water blocks. Each
    query is a single RPC, with no hidden inner executor. State includes queues,
    source cursors and uncompleted operations for crash-safe replay from caches.
    """
    state = state or {}
    restored = state.get("candidates", [])
    replay = deque(state.get("pending_queries", []))
    planner = SamplePlanner(
        [*candidates, *restored, *state.get("pending_samples", [])], dates, ledger, rng, tried
    )
    pending, refilling, exhausted = {}, set(), set()
    sources = {"water": water_source, "ice": ice_source}
    source_turn, last_checkpoint = 0, 0.0

    def stopped():
        return stop is not None and stop.is_set()

    def save(force=False):
        nonlocal last_checkpoint
        if checkpoint is None or (not force and time.monotonic() - last_checkpoint < 15):
            return
        checkpoint(
            dict(
                version=1,
                candidates=planner.snapshot(),
                pending_samples=[item for kind, item in pending.values() if kind == "sample"],
                pending_queries=[list(pair) for pair in pending.values() if pair[0] != "sample"]
                + list(replay),
                water=water_source.snapshot() if water_source is not None else {},
                ice=ice_source.snapshot() if ice_source is not None else {},
            )
        )
        LOG.info(
            "Candidate buffers: %s; active=%d",
            {g: planner.count({g}) for g in target},
            len(pending),
        )
        last_checkpoint = time.monotonic()

    try:
        while True:
            reserved = SamplingLedger(
                [item for kind, item in pending.values() if kind == "sample"],
                ledger.max_dates,
                ledger.min_days,
            )
            remaining = {
                g: max(0, n - ledger.counts[g] - reserved.counts[g]) for g, n in target.items()
            }
            limits = {
                g: min(ice_buffer if g == ICE_GROUP else capacity, 3 * n)
                for g, n in remaining.items()
            }
            planner.trim({g: ice_buffer if g == ICE_GROUP else capacity for g in target})
            for group, limit in limits.items():
                count = planner.count({group})
                low = ice_low_watermark if group == ICE_GROUP else low_watermark
                if not limit or count >= limit:
                    refilling.discard(group)
                elif count <= min(low, limit - 1):
                    refilling.add(group)

            def query_slot():
                nonlocal source_turn
                # next() may discard many blocked centres since the outer check.
                # Recompute low-water flags before deciding the source is empty.
                for group, total in target.items():
                    left = max(0, total - ledger.counts[group] - reserved.counts[group])
                    limit = min(ice_buffer if group == ICE_GROUP else capacity, 3 * left)
                    count = planner.count({group})
                    low = ice_low_watermark if group == ICE_GROUP else low_watermark
                    if not limit or count >= limit:
                        refilling.discard(group)
                    elif count <= min(low, limit - 1):
                        refilling.add(group)
                choices = []
                needs = {g for g in refilling if ledger.counts[g] + reserved.counts[g] < target[g]}
                if any(g != ICE_GROUP for g in needs):
                    choices.append("water")
                if ICE_GROUP in needs:
                    choices.append("ice")
                if len(choices) == 2 and source_turn % 2:
                    choices.reverse()
                for kind in choices:
                    source = sources[kind]
                    if source is None or kind in exhausted:
                        continue
                    if kind == "ice" and any(k == "ice" for k, _ in pending.values()):
                        continue
                    query = None
                    for pair in list(replay):
                        if pair[0] == kind:
                            replay.remove(pair)
                            query = pair[1]
                            break
                    if query is None:
                        query = source.next_query(ledger, reserved)
                    if query is None:
                        # Ice hints may become available after a sample finishes.
                        if kind == "water":
                            exhausted.add(kind)
                        continue
                    pending[workers.submit(source.fetch, query)] = (kind, query)
                    source_turn += 1
                    return True
                return False

            if not stopped():
                # Reserve at most one refill slot when sample work is available.
                if len(pending) < concurrency and not any(
                    k != "sample" for k, _ in pending.values()
                ):
                    query_slot()
                while len(pending) < concurrency:
                    sample = planner.next(ledger, reserved, target)
                    if sample is not None:
                        pending[workers.submit(collect_one, sample)] = ("sample", sample)
                        reserved.add(sample)
                    elif not query_slot():
                        break
            save()
            if not pending:
                return
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                kind, item = pending[future]
                # Leave failures in the checkpoint so a restart retries them.
                result = future.result()
                pending.pop(future)
                if kind != "sample":
                    planner.extend(result, ledger, limits)
                    continue
                if result is not None:
                    ledger.add(result)
                    if ice_source is not None:
                        ice_source.observe(result)
                yield result
    finally:
        save(force=True)
        for future in pending:
            future.cancel()
