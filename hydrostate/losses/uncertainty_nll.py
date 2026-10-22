"""Masked Gaussian likelihood with one learned residual variance per task."""

from __future__ import annotations

import math

import torch
import torch.distributed as dist
from torch import nn

from hydrostate.normalization import load_target_statistics
from hydrostate.registry import MODELS
from hydrostate.tasks import LEGACY_TASKS, resolve_tasks


def global_observation_means(local_sums, counts, count_indices):
    """Reduce head sums and unique task counts in one DDP collective.

    Final and auxiliary heads index the same denominator. Gradients compensate
    for DDP averaging; reported losses are global valid-observation means.
    """
    totals = torch.cat((torch.stack([value.detach() for value in local_sums]),
                        counts.to(local_sums[0])))
    world_size = 1
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
        dist.all_reduce(totals)
    global_counts = totals[len(local_sums):]
    denominator = global_counts.clamp(min=1).index_select(0, count_indices)
    global_mean = totals[:len(local_sums)] / denominator
    values = []
    for index, local_sum in enumerate(local_sums):
        objective = world_size * local_sum / denominator[index]
        values.append(objective + (global_mean[index] - objective.detach()))
    # Do not stack differentiable sums: an auxiliary-only backward must not
    # materialize zero gradients on final heads or learned final-loss variances.
    return values, global_counts


def global_observation_mean(local_sum, count):
    """Scalar compatibility wrapper, with no host read of a CUDA count."""
    count = torch.as_tensor(count, device=local_sum.device, dtype=local_sum.dtype)
    values, counts = global_observation_means(
        [local_sum], count.reshape(1),
        torch.zeros(1, dtype=torch.long, device=local_sum.device),
    )
    return values[0], counts[0]


def _masked_inputs(prediction, target, valid):
    prediction, target, valid = torch.broadcast_tensors(prediction, target, valid.bool())
    # Sanitize before arithmetic: multiplying a NaN loss by zero is insufficient.
    return prediction.float().masked_fill(~valid, 0), target.float().masked_fill(~valid, 0), valid


def masked_standardized_mse_sum(prediction, target, valid, std):
    prediction, target, valid = _masked_inputs(prediction, target, valid)
    return ((prediction - target) / std).square().masked_fill(~valid, 0).sum()


def masked_standardized_mse(prediction, target, valid, std):
    """Fixed-scale auxiliary objective, including empty/NaN and DDP handling."""
    prediction, target, valid = torch.broadcast_tensors(prediction, target, valid.bool())
    local_sum = masked_standardized_mse_sum(prediction, target, valid, std)
    return global_observation_mean(local_sum, valid.sum())


@MODELS.register_module()
class UncertaintyWeightedNLL(nn.Module):
    """Standardize physical observations, then average each task's Gaussian NLL.

    Task variances are homoscedastic, in standardized units. A task's mean is
    taken over all valid observations in the global DDP microbatch. Multiplying
    local gradients by world size compensates for DDP's gradient averaging.
    """

    def __init__(self, target_statistics: str | dict, min_variance: float = 1e-6, tasks=LEGACY_TASKS):
        super().__init__()
        if not math.isfinite(min_variance) or not 0 < min_variance < 1:
            raise ValueError("min_variance must be finite and in (0, 1)")
        self.min_variance = min_variance
        self.tasks = tuple(resolve_tasks(tasks))
        self.target_statistics = load_target_statistics(target_statistics, tasks)
        values = self.target_statistics["tasks"]
        self.register_buffer("target_mean", torch.tensor([values[t]["mean"] for t in self.tasks]))
        self.register_buffer("target_std", torch.tensor([values[t]["std"] for t in self.tasks]))
        self.register_buffer(
            "training_fingerprint",
            torch.tensor(
                list(bytes.fromhex(self.target_statistics["training_fingerprint"])),
                dtype=torch.uint8,
            ),
        )
        self.log_variances = nn.ParameterDict(
            {task: nn.Parameter(torch.zeros(())) for task in self.tasks}
        )

    def variance(self, task):
        # The floor avoids a singular likelihood for nearly perfect predictions.
        return self.log_variances[task].float().clamp(min=math.log(self.min_variance)).exp()

    def forward(self, task, prediction, target, valid):
        prediction, target, valid = torch.broadcast_tensors(prediction, target, valid.bool())
        return global_observation_mean(self.local_sum(task, prediction, target, valid), valid.sum())

    def local_sum(self, task, prediction, target, valid, weight=None):
        """Fixed-shape NLL sum; leave shared normalization to the caller."""
        index = self.tasks.index(task)
        prediction, target, valid = _masked_inputs(prediction, target, valid)
        mean, std = self.target_mean[index], self.target_std[index]
        standardized_prediction = (prediction - mean) / std
        standardized_target = (target - mean) / std
        variance = self.variance(task)
        # Match GaussianNLLLoss's clamp, ignored by autograd. The exponential
        # already ensures positivity, avoiding a Python/CUDA validity check.
        variance = variance.clone()
        with torch.no_grad():
            variance.clamp_(min=self.min_variance)
        value = 0.5 * (
            variance.log() + (standardized_prediction - standardized_target).square() / variance
        ) + 0.5 * math.log(2 * math.pi)
        # Also mask the normalizing constant and variance penalty for missing data.
        if weight is not None:
            value = value * weight.detach().float().masked_fill(~valid, 0)
        return value.masked_fill(~valid, 0).sum()

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        for name in ("target_mean", "target_std", "training_fingerprint"):
            key = prefix + name
            if key not in state_dict or not torch.equal(
                state_dict[key].cpu(), getattr(self, name).cpu()
            ):
                raise ValueError(f"Checkpoint normalization disagrees with configuration: {key}")
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
