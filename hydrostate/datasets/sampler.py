"""Balanced distributed sampling from manifest-defined sample weights."""

from __future__ import annotations

import math

import torch
from mmengine.dist import get_dist_info, sync_random_seed
from torch.utils.data import Sampler

from hydrostate.registry import DATA_SAMPLERS


@DATA_SAMPLERS.register_module()
class HydroBalancedSampler(Sampler[int]):
    """Weighted sampler with deterministic disjoint streams for distributed jobs."""

    def __init__(
        self,
        dataset,
        shuffle: bool = True,
        seed: int | None = None,
        round_up: bool = True,
    ) -> None:
        self.dataset = dataset
        self.shuffle = shuffle
        self.rank, self.world_size = get_dist_info()
        self.seed = sync_random_seed() if seed is None else seed
        self.epoch = 0
        self.num_samples = math.ceil(len(dataset) / self.world_size) if round_up else len(
            range(self.rank, len(dataset), self.world_size)
        )
        self.total_size = self.num_samples * self.world_size

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        if self.shuffle:
            weights = self.dataset.sampling_weights()
            indices = torch.multinomial(
                weights, self.total_size, replacement=True, generator=generator
            ).tolist()
        else:
            indices = list(range(len(self.dataset)))
            if len(indices) < self.total_size:
                indices.extend(indices[: self.total_size - len(indices)])
            else:
                indices = indices[: self.total_size]
        return iter(indices[self.rank : self.total_size : self.world_size])

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
