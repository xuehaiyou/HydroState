"""Uniform no-replacement epochs with product and station coverage in each batch."""

import numpy as np
from torch.utils.data import Sampler

from hydrostate.registry import DATA_SAMPLERS


@DATA_SAMPLERS.register_module()
class FullSourceSampler(Sampler):
    def __init__(self, dataset, seed=3407, batch_size=10):
        self.n = len(dataset)
        self.batch_size, self.seed, self.epoch = batch_size, seed, 0
        self.product = np.arange(len(dataset.product))
        self.manual = np.arange(dataset.base_length, self.n)
        self.sm, self.et = [
            np.concatenate(
                [
                    dataset.offsets[i] + np.flatnonzero(site.table["valid_" + task].to_numpy() > 0)
                    for i, site in enumerate(dataset.sites, start=1)
                ]
            )
            for task in ("sm", "et")
        ]
        ids = np.concatenate([self.product, self.sm, self.et, self.manual])
        if len(ids) != self.n or len(np.unique(ids)) != self.n:
            raise ValueError("Expected disjoint, exhaustive product, SM, ET and manual cohorts")

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.n

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        full, tail = divmod(self.n, self.batch_size)
        capacities = [self.batch_size] * full + ([tail] if tail else [])
        bins = [[] for _ in capacities]
        pools = [rng.permutation(v).tolist() for v in (self.product, self.sm, self.manual)]
        for batch, capacity in zip(bins, capacities):
            for pool in pools[: min(3, capacity)]:
                batch.append(pool.pop())
        # Spread sparse ET across different full batches before filling remaining slots.
        et = rng.permutation(self.et).tolist()
        for i in rng.permutation(full)[: len(et)]:
            bins[i].append(et.pop())
        spare = rng.permutation([i for pool in [*pools, et] for i in pool]).tolist()
        for batch, capacity in zip(bins, capacities):
            batch.extend(spare.pop() for _ in range(capacity - len(batch)))
            rng.shuffle(batch)
            yield from batch
