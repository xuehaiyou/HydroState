"""MMEngine joint HydroState retrieval model."""

from __future__ import annotations

from typing import Any

import torch
from mmengine.model import BaseModel

from hydrostate.registry import MODELS

TASKS = ("water", "soil_moisture", "et")


@MODELS.register_module()
class HydroStateModel(BaseModel):
    """One OlmoEarth encoder, one shared decoder, and three retrieval heads."""

    def __init__(
        self,
        encoder: dict[str, Any],
        decoder: dict[str, Any],
        heads: dict[str, dict[str, Any]],
        loss: dict[str, Any],
        task_weights: dict[str, float] | None = None,
        site_loss_weight: float = 3.0,
        data_preprocessor: dict[str, Any] | None = None,
        init_cfg: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        if set(heads) != set(TASKS):
            raise ValueError(f"heads must be exactly {TASKS}, got {tuple(heads)}")
        self.encoder = MODELS.build(encoder)
        self.decoder = MODELS.build(decoder)
        self.heads = torch.nn.ModuleDict({name: MODELS.build(cfg) for name, cfg in heads.items()})
        self.loss_module = MODELS.build(loss)
        self.task_weights = task_weights or {name: 1.0 for name in TASKS}
        self.site_loss_weight = site_loss_weight

    def _forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        features = self.encoder(inputs)
        decoded = self.decoder(features)
        return {name: self.heads[name](decoded) for name in TASKS}

    def loss(
        self, inputs: dict[str, torch.Tensor], data_samples: dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        outputs = self._forward(inputs)
        losses: dict[str, torch.Tensor] = {}
        for task in TASKS:
            task_loss = self.loss_module(
                outputs[task], data_samples[task], data_samples[f"{task}_valid"]
            )
            losses[f"loss_{task}"] = self.task_weights[task] * task_loss

        if all(key in data_samples for key in ("site_value", "site_valid", "site_footprint")):
            footprint = data_samples["site_footprint"].float()
            if footprint.ndim == 3:
                footprint = footprint[:, None]
            denominator = footprint.sum(dim=(-2, -1)).clamp(min=1e-6)
            site_values = data_samples["site_value"].float()
            site_valid = data_samples["site_valid"].bool()
            for task_index, task in enumerate(TASKS):
                pooled = (outputs[task] * footprint).sum(dim=(-2, -1)) / denominator
                target = site_values[:, task_index].reshape(-1, 1)
                valid = site_valid[:, task_index].reshape(-1, 1)
                losses[f"loss_{task}_site"] = (
                    self.site_loss_weight
                    * self.task_weights[task]
                    * self.loss_module(pooled, target, valid)
                )
        return losses

    def predict(
        self, inputs: dict[str, torch.Tensor], data_samples: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        outputs = self._forward(inputs)
        batch_size = next(iter(outputs.values())).shape[0]
        predictions: list[dict[str, Any]] = []
        metainfo = {} if data_samples is None else data_samples.get("metainfo", {})
        for index in range(batch_size):
            item = {task: outputs[task][index] for task in TASKS}
            item["metainfo"] = {
                key: value[index] if isinstance(value, (list, tuple)) else value
                for key, value in metainfo.items()
            }
            if data_samples is not None:
                for task in TASKS:
                    item[f"target_{task}"] = data_samples[task][index]
                    item[f"valid_{task}"] = data_samples[f"{task}_valid"][index]
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
