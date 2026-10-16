"""One task/time-query module for every predicted variable.

The interface uses UTC epoch-second bounds, [t,t] for an instant and [start,end)
for a retrieval interval. It has no product-specific or LST-specific branches.
"""

import math

import torch
from torch import nn

from hydrostate.registry import MODELS


@MODELS.register_module()
class TargetTimeQuery(nn.Module):
    """A task/time query selects valid observations independently at each patch."""

    def __init__(self, channels, tasks, query_channels=64):
        super().__init__()
        self.tasks = tuple(tasks)
        self.keys = nn.Sequential(nn.LayerNorm(channels), nn.Linear(channels, query_channels))
        self.task_embedding = nn.Embedding(len(tasks), query_channels)
        self.time_query = nn.Sequential(nn.Linear(5, query_channels), nn.GELU(),
                                        nn.Linear(query_channels, query_channels))
        self.relative_score = nn.Sequential(nn.Linear(3, query_channels), nn.GELU(),
                                            nn.Linear(query_channels, 1))
        self.query_value = nn.Linear(query_channels, channels)
        self.scale = query_channels ** -.5

    def forward(self, sequence, bounds, task, keys=None):
        values, valid, times = sequence["features"], sequence["valid"], sequence["time"]
        if bounds.shape != (values.shape[0], 2):
            raise ValueError("Target times must be B,2 UTC epoch-second bounds")
        if bounds.dtype != torch.int64:
            raise ValueError("Target times must use int64 UTC epoch seconds")
        torch._assert_async(((bounds[:, 0] > 0) & (bounds[:, 1] >= bounds[:, 0])).all(),
                            "Invalid target time bounds")
        bounds = bounds.to(device=values.device, dtype=torch.float64)
        center = bounds.mean(1)
        # Annual season and UTC time-of-day are independent of relative image age.
        annual = center / (86400 * 365.2425) * (2 * math.pi)
        daily = center / 86400 * (2 * math.pi)
        duration = (bounds[:, 1] - bounds[:, 0]) / 86400
        absolute = torch.stack((annual.sin(), annual.cos(), daily.sin(), daily.cos(),
                                duration.log1p() / math.log(65)), dim=-1).float()
        query = self.task_embedding.weight[self.tasks.index(task)] + self.time_query(absolute)
        if keys is None:
            keys = self.keys(values.permute(0, 1, 3, 4, 2))
        logits = (keys * query[:, None, None, None, :]).sum(-1).float() * self.scale
        # Use seconds until after subtraction to retain sub-day acquisition offsets.
        times = times.to(device=values.device, dtype=torch.float64)
        relative = torch.stack(((times - bounds[:, :1]) / (64 * 86400),
                                (times - bounds[:, 1:]) / (64 * 86400),
                                (times - center[:, None]).abs() / (64 * 86400)), dim=-1)
        relative_logits = self.relative_score(relative.clamp(-4, 4).float())[..., 0]
        logits = logits + relative_logits[:, :, None, None].float()
        weights = logits.masked_fill(~valid, -torch.finfo(logits.dtype).max).softmax(dim=1)
        weights = weights * valid
        weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-8)
        pooled = (values.masked_fill(~valid[:, :, None], 0) * weights[:, :, None]).sum(dim=1)
        # A residual query also conditions the one-observation case, where softmax=1.
        pooled = pooled + self.query_value(query)[:, :, None, None]
        return pooled.masked_fill(~valid.any(dim=1)[:, None], 0)
