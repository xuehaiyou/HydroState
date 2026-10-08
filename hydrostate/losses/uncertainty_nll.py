"""Masked Gaussian likelihood with one learned residual variance per task."""

from __future__ import annotations

import math

import torch
import torch.distributed as dist
from torch import nn

from hydrostate.normalization import TASKS, load_target_statistics
from hydrostate.registry import MODELS


@MODELS.register_module()
class UncertaintyWeightedNLL(nn.Module):
    """Standardize physical observations, then average each task's Gaussian NLL.

    Task variances are homoscedastic, in standardized units. A task's mean is
    taken over all valid observations in the global DDP microbatch. Multiplying
    local gradients by world size compensates for DDP's gradient averaging.
    """

    def __init__(self, target_statistics: str | dict, min_variance: float = 1e-6):
        super().__init__()
        if not math.isfinite(min_variance) or not 0 < min_variance < 1:
            raise ValueError("min_variance must be finite and in (0, 1)")
        self.min_variance = min_variance
        self.target_statistics = load_target_statistics(target_statistics)
        values = self.target_statistics["tasks"]
        self.register_buffer("target_mean", torch.tensor([values[t]["mean"] for t in TASKS]))
        self.register_buffer("target_std", torch.tensor([values[t]["std"] for t in TASKS]))
        self.register_buffer(
            "training_fingerprint",
            torch.tensor(
                list(bytes.fromhex(self.target_statistics["training_fingerprint"])),
                dtype=torch.uint8,
            ),
        )
        self.log_variances = nn.ParameterDict(
            {task: nn.Parameter(torch.zeros(())) for task in TASKS}
        )

    def variance(self, task):
        # The floor avoids a singular likelihood for nearly perfect predictions.
        return self.log_variances[task].float().clamp(min=math.log(self.min_variance)).exp()

    def forward(self, task, prediction, target, valid):
        index = TASKS.index(task)
        prediction, target, valid = torch.broadcast_tensors(prediction, target, valid.bool())
        # Select before arithmetic: NaNs in invalid labels must not poison gradients.
        prediction = prediction.float().masked_select(valid)
        target = target.float().masked_select(valid)
        mean, std = self.target_mean[index], self.target_std[index]
        standardized_prediction = (prediction - mean) / std
        standardized_target = (target - mean) / std
        variance = self.variance(task)
        local_sum = torch.nn.functional.gaussian_nll_loss(
            standardized_prediction,
            standardized_target,
            variance.expand_as(prediction),
            full=True,
            eps=self.min_variance,
            reduction="sum",
        )
        # Empty selections still connect both the head and log variance to autograd.
        totals = torch.stack((local_sum.detach(), local_sum.new_tensor(prediction.numel())))
        world_size = 1
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            dist.all_reduce(totals)
        denominator = totals[1].clamp(min=1)
        objective = world_size * local_sum / denominator
        global_mean = totals[0] / denominator
        # Log the same true global mean on every rank, preserving local gradients.
        objective = objective + (global_mean - objective.detach())
        return objective, totals[1]

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        for name in ("target_mean", "target_std", "training_fingerprint"):
            key = prefix + name
            if key not in state_dict or not torch.equal(
                state_dict[key].cpu(), getattr(self, name).cpu()
            ):
                raise ValueError(f"Checkpoint normalization disagrees with configuration: {key}")
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
