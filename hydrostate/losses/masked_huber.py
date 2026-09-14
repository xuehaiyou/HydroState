"""Masked regression losses with support for coarse product labels."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from hydrostate.registry import MODELS


def _as_bchw(value: torch.Tensor) -> torch.Tensor:
    if value.ndim == 1:
        return value[:, None, None, None]
    if value.ndim == 2:
        return value[:, :, None, None]
    if value.ndim == 3:
        return value[:, None]
    if value.ndim != 4:
        raise ValueError(f"expected a B, BxC, BxHxW or BxCxHxW tensor, got {value.shape}")
    return value


@MODELS.register_module()
class MaskedHuberLoss(nn.Module):
    """Huber loss evaluated only where target labels are valid.

    If a product target is coarser than the prediction, predictions are area
    pooled to the target shape before comparison. This avoids pretending a
    coarse product is a native 10 m label.
    """

    def __init__(self, delta: float = 1.0, loss_weight: float = 1.0) -> None:
        super().__init__()
        self.delta = delta
        self.loss_weight = loss_weight

    def forward(
        self, prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        prediction = _as_bchw(prediction)
        target = _as_bchw(target).to(dtype=prediction.dtype)
        valid = _as_bchw(valid).bool()
        if prediction.shape[-2:] != target.shape[-2:]:
            prediction = F.adaptive_avg_pool2d(prediction, target.shape[-2:])
        if valid.shape[-2:] != target.shape[-2:]:
            valid = F.adaptive_max_pool2d(valid.float(), target.shape[-2:]).bool()
        valid = valid.expand_as(target)
        count = valid.sum()
        if count == 0:
            # Retain a differentiable zero so distributed loss parsing is stable.
            return prediction.sum() * 0.0
        error = F.huber_loss(prediction, target, reduction="none", delta=self.delta)
        return self.loss_weight * error.masked_select(valid).mean()
