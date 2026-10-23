"""Small, shuffled centre pools, built from cached blocks before pixel downloads."""

import logging
from collections import Counter

from .sampling import location_key

LOG = logging.getLogger(__name__)


def candidate_pools(blocks, ledger, target, rng, factor=3):
    """Fill unfinished strata to factor times their remaining sample quotas.

    Only lightweight coordinates/metadata are retained. Each block batch mixes
    shuffled global regions. Full strata retain no additional centres; source
    blocks remain cached for later passes. This is not a global pixel census.
    """
    source, exhausted, seen = iter(blocks), False, set()
    while not exhausted:
        capacity = {k: factor * max(0, n - ledger.counts[k]) for k, n in target.items()}
        if not any(capacity.values()):
            return
        pool, counts, batches = [], Counter(), 0
        while any(counts[k] < n for k, n in capacity.items()):
            try:
                batch = list(next(source))
            except StopIteration:
                exhausted = True
                break
            batches += 1
            rng.shuffle(batch)
            for sample in batch:
                key, fb = location_key(sample), sample["frequency_bin"]
                if (
                    counts[fb] >= capacity.get(fb, 0)
                    or key in seen
                    or len(ledger.dates.get(key, ())) >= ledger.max_dates
                ):
                    continue
                seen.add(key)
                pool.append(sample)
                counts[fb] += 1
            LOG.info("Building candidate pool: counts=%s; targets=%s", dict(counts), capacity)
        if pool:
            rng.shuffle(pool)
            LOG.info("Candidate pool ready: %d centres from %d block batches", len(pool), batches)
            yield pool
