"""MMEngine joint HydroState retrieval model."""

from __future__ import annotations

from typing import Any

import torch
from mmengine.model import BaseModel

from hydrostate.normalization import TASKS
from hydrostate.registry import MODELS


def supervision_values(task, prediction, samples):
    """Return predictions and labels on the same observational support."""
    prediction = prediction.float()  # Stable reductions under BF16 inference/training.
    support = samples.get("prediction_valid", torch.ones_like(prediction, dtype=torch.bool))
    if task == "water":
        return prediction, samples[task], samples[f"{task}_valid"] & support
    if task == "soil_moisture":
        count = support.sum(dim=(-2, -1), keepdim=True)
        pooled = prediction.masked_fill(~support, 0).sum(dim=(-2, -1), keepdim=True)
        pooled = pooled / count.clamp(min=1)
        return pooled, samples[task], samples[f"{task}_valid"] & (count > 0)
    if "et_native_weights" not in samples:
        raise ValueError("ET supervision requires native pixel area weights")
    weights = samples["et_native_weights"].float()
    pooled = (prediction * weights).sum(dim=(-2, -1))
    complete = ~((weights > 0) & ~support).any(dim=(-2, -1))
    valid = samples["et_native_valid"] & complete & (weights.sum(dim=(-2, -1)) > 0)
    return pooled, samples["et_native_value"], valid


def site_supervision_values(task, prediction, samples):
    """Optional station observation on a completely observed, nonempty footprint."""
    if not all(key in samples for key in ("site_value", "site_valid", "site_footprint")):
        return None
    footprint = samples["site_footprint"].float()
    if footprint.ndim == 3:
        footprint = footprint[:, None]
    denominator = footprint.sum(dim=(-2, -1))
    pooled = (prediction.float() * footprint).sum(dim=(-2, -1)) / denominator.clamp(min=1e-6)
    index = TASKS.index(task)
    target = samples["site_value"][:, index].reshape(-1, 1)
    valid = samples["site_valid"][:, index].bool().reshape(-1, 1) & (denominator > 0)
    if "prediction_valid" in samples:
        valid = valid & ~((footprint > 0) & ~samples["prediction_valid"]).any(dim=(-2, -1))
    return pooled, target, valid


@MODELS.register_module()
class HydroStateModel(BaseModel):
    """One OlmoEarth encoder, one shared decoder, and three retrieval heads."""

    def __init__(
        self,
        encoder: dict[str, Any],
        decoder: dict[str, Any],
        heads: dict[str, dict[str, Any]],
        loss: dict[str, Any],
        target_statistics: str | dict,
        output_scale: int = 3,
        data_preprocessor: dict[str, Any] | None = None,
        init_cfg: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        if set(heads) != set(TASKS):
            raise ValueError(f"heads must be exactly {TASKS}, got {tuple(heads)}")
        self.encoder = MODELS.build(encoder)
        self.decoder = MODELS.build(decoder)
        self.heads = torch.nn.ModuleDict({name: MODELS.build(cfg) for name, cfg in heads.items()})
        self.loss_module = MODELS.build(dict(loss, target_statistics=target_statistics))
        self.output_scale = output_scale

    def _forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        features = self.encoder(inputs)
        decoded = self.decoder(features)
        if decoded.shape[-1] % self.output_scale or decoded.shape[-2] % self.output_scale:
            raise ValueError("Decoded grid must be divisible by output_scale")
        decoded = torch.nn.functional.avg_pool2d(decoded, self.output_scale)
        return {name: self.heads[name](decoded) for name in TASKS}

    def loss(
        self, inputs: dict[str, torch.Tensor], data_samples: dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        outputs = self._forward(inputs)
        losses: dict[str, torch.Tensor] = {}
        for task in TASKS:
            prediction, target, valid = supervision_values(task, outputs[task], data_samples)
            site = site_supervision_values(task, outputs[task], data_samples)
            if site is not None:
                prediction, target, valid = (
                    torch.cat((product.flatten(), station.flatten()))
                    for product, station in zip((prediction, target, valid), site)
                )
            task_loss, count = self.loss_module(task, prediction, target, valid)
            losses[f"loss_{task}"] = task_loss
            losses[f"valid_count_{task}"] = count
            variance = self.loss_module.variance(task).detach()
            losses[f"variance_{task}"] = variance
            losses[f"precision_{task}"] = variance.reciprocal()
        return losses

    def predict(
        self, inputs: dict[str, torch.Tensor], data_samples: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        outputs = self._forward(inputs)
        batch_size = next(iter(outputs.values())).shape[0]
        predictions: list[dict[str, Any]] = []
        metainfo = {} if data_samples is None else data_samples.get("metainfo", {})
        supervised = (
            {}
            if data_samples is None
            else {task: supervision_values(task, outputs[task], data_samples) for task in TASKS}
        )
        for index in range(batch_size):
            item = {task: outputs[task][index] for task in TASKS}
            item["metainfo"] = {
                key: value[index] if isinstance(value, (list, tuple)) else value
                for key, value in metainfo.items()
            }
            if data_samples is not None:
                # Preserve spatial support for validation visualization. ET
                # targets/predictions below are scalars on native footprints.
                for key in ("prediction_valid", "et_native_weights"):
                    if key in data_samples:
                        item[key] = data_samples[key][index]
                for task in TASKS:
                    pred, target, valid = supervised[task]
                    item[f"supervised_{task}"] = pred[index]
                    item[f"target_{task}"] = target[index]
                    item[f"valid_{task}"] = valid[index]
            predictions.append(item)
        return predictions

    def forward(
        self,
        inputs: dict[str, torch.Tensor],
        data_samples: dict[str, Any] | None = None,
        mode: str = "tensor",
    ):
        if mode == "loss":
            if data_samples is None:
                raise ValueError("data_samples are required for loss mode")
            return self.loss(inputs, data_samples)
        if mode == "predict":
            return self.predict(inputs, data_samples)
        if mode == "tensor":
            return self._forward(inputs)
        raise RuntimeError(f"unsupported mode {mode!r}")
