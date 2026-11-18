"""PLE equations (2)--(7): task gates plus an evolving shared stream.

The recommendation MLP is applied pointwise to spatial features. Unlike the
LibMTL ResNet variant, intermediate shared gates mix ALL current experts.
"""

import torch
from torch import nn

from hydrostate.registry import MODELS

from .state import CoSAStateFusion


class ExtractionLevel(nn.Module):
    def __init__(self, tasks, in_channels, channels, num_shared, num_private, final):
        super().__init__()
        self.tasks = tuple(tasks)
        self.private = nn.ModuleDict({task: nn.ModuleList([
            nn.Sequential(nn.Conv2d(in_channels, channels, 1), nn.ReLU())
            for _ in range(num_private)]) for task in tasks})
        self.shared = nn.ModuleList([
            nn.Sequential(nn.Conv2d(in_channels, channels, 1), nn.ReLU())
            for _ in range(num_shared)])
        self.gates = nn.ModuleDict({task: nn.Conv2d(
            in_channels, num_private + num_shared, 1) for task in tasks})
        self.shared_gate = None if final else nn.Conv2d(
            in_channels, len(tasks) * num_private + num_shared, 1)

    @staticmethod
    def mix(gate, selector, candidates, valid):
        values = torch.stack(candidates, dim=1)
        weights = gate(selector).float().softmax(1).to(values.dtype)
        return (values * weights[:, :, None]).sum(1).masked_fill(~valid, 0)

    def forward(self, streams, shared, valid):
        shared = shared.masked_fill(~valid, 0)
        private = {task: [expert(streams[task].masked_fill(~valid, 0)).masked_fill(~valid, 0)
                          for expert in self.private[task]] for task in self.tasks}
        common = [expert(shared).masked_fill(~valid, 0) for expert in self.shared]
        # All updates use the previous level: no task-order dependent mutation.
        result = {task: self.mix(self.gates[task], streams[task],
                                 private[task] + common, valid) for task in self.tasks}
        next_shared = None
        if self.shared_gate is not None:
            candidates = [value for task in self.tasks for value in private[task]] + common
            next_shared = self.mix(self.shared_gate, shared, candidates, valid)
        return result, next_shared


class ProgressiveExtraction(nn.Module):
    def __init__(self, tasks, in_channels=256, widths=(256, 128),
                 num_shared=1, num_private=1, tower_channels=64):
        super().__init__()
        self.tasks = tuple(tasks)
        if (not self.tasks or len(set(self.tasks)) != len(self.tasks) or not widths
                or min(in_channels, *widths, num_shared, num_private, tower_channels) < 1):
            raise ValueError('Invalid PLE dimensions or tasks')
        self.levels = nn.ModuleList([
            ExtractionLevel(self.tasks, in_channels if i == 0 else widths[i - 1], width,
                            num_shared, num_private, final=i == len(widths) - 1)
            for i, width in enumerate(widths)])
        self.towers = nn.ModuleDict({task: nn.Sequential(
            nn.Conv2d(widths[-1], tower_channels, 1), nn.ReLU()) for task in self.tasks})

    def forward(self, value, valid):
        streams = {task: value for task in self.tasks}
        shared = value
        for level in self.levels:
            streams, shared = level(streams, shared, valid)
        return {task: self.towers[task](streams[task]).masked_fill(~valid, 0)
                for task in self.tasks}


@MODELS.register_module()
class PLEDecoder(nn.Module):
    def __init__(self, groups, temporal=None, widths=(256, 128), num_shared=1,
                 num_private=1, tower_channels=64):
        super().__init__()
        self.groups = tuple(groups)
        self.temporal = CoSAStateFusion(**(temporal or {}))
        self.extraction = ProgressiveExtraction(
            self.groups, self.temporal.out_channels, widths, num_shared, num_private,
            tower_channels)
        self.out_channels = tower_channels

    def prepare(self, levels, inputs):
        return self.temporal.prepare(levels, inputs)

    def forward(self, prepared, bounds):
        value, valid = self.temporal(prepared, bounds)
        return self.extraction(value, valid), valid, None
