"""Bounded rolling workers with reservations for hard sampling constraints."""

from collections import deque
from concurrent.futures import FIRST_COMPLETED, wait

from .sampling import SamplingLedger, location_key


def completed_samples(workers, proposals, ledger, target, concurrency, collect_one):
    """Commit in completion order; failed eligibility releases its reservation.

    Only the calling thread touches the ledger. Temporarily blocked proposals
    stay queued so a rejected neighbour cannot silently consume a quota.
    """
    proposals, pending = deque(proposals), {}
    try:
        while proposals or pending:
            reserved = SamplingLedger(pending.values(), ledger.max_dates, ledger.min_days)
            for _ in range(len(proposals)):
                if len(pending) >= concurrency:
                    break
                sample = proposals.popleft()
                fb, day = sample["frequency_bin"], sample["target_time"]
                if ledger.counts[fb] >= target[fb] or not ledger.allows(sample, day):
                    continue
                key = location_key(sample)
                if (
                    ledger.counts[fb] + reserved.counts[fb] >= target[fb]
                    or len(ledger.dates[key]) + len(reserved.dates[key]) >= ledger.max_dates
                    or not reserved.allows(sample, day)
                ):
                    proposals.append(sample)
                    continue
                pending[workers.submit(collect_one, sample)] = sample
                reserved.add(sample)
            if not pending:
                break
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                pending.pop(future)
                # Network failures propagate; only explicit ineligibility is None.
                row = future.result()
                if row is not None:
                    ledger.add(row)
                yield row
    finally:
        for future in pending:
            future.cancel()
