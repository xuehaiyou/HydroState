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

    Spatial support must be aligned explicitly before calling this loss.
    Tensor dimensions alone do not identify a product's physical footprint.
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
        if prediction.shape != target.shape or valid.shape != target.shape:
            raise ValueError(
                "Prediction, target and mask must have identical observational support"
            )
        valid = valid.expand_as(target)
        count = valid.sum()
        if count == 0:
            # Retain a differentiable zero so distributed loss parsing is stable.
            return prediction.sum() * 0.0
        return self.loss_weight * F.huber_loss(
            prediction.masked_select(valid), target.masked_select(valid), delta=self.delta
        )
