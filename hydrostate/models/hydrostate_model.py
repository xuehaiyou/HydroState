"""MMEngine joint HydroState retrieval model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from mmengine.model import BaseModel

from hydrostate.losses.uncertainty_nll import (
    global_observation_means,
    masked_standardized_mse_sum,
)
from hydrostate.normalization import TASKS
from hydrostate.registry import MODELS

from .water_soil_coupling import WaterSoilCoupling, weighted_sm_mean


def supervision_values(task, prediction, samples, sm_support=None):
    """Return predictions and labels on the same observational support."""
    prediction = prediction.float()  # Stable reductions under BF16 inference/training.
    support = samples.get("prediction_valid", torch.ones_like(prediction, dtype=torch.bool))
    if task == "water":
        return prediction, samples[task], samples[f"{task}_valid"] & support
    if task == "soil_moisture":
        if sm_support is not None:
            return (sm_support.project(prediction), samples[task],
                    samples[f"{task}_valid"] & sm_support.valid)
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


def site_supervision_values(task, prediction, samples, sm_support=None):
    """Optional station observation on a completely observed, nonempty footprint."""
    if not all(key in samples for key in ("site_value", "site_valid", "site_footprint")):
        return None
    footprint = samples["site_footprint"].float()
    if footprint.ndim == 3:
        footprint = footprint[:, None]
    if task == "soil_moisture" and sm_support is not None:
        weights, denominator, complete = sm_support.station_support(footprint)
        pooled = weighted_sm_mean(prediction, weights, denominator[..., None, None])
        index = TASKS.index(task)
        target = samples["site_value"][:, index].reshape(-1, 1)
        valid = samples["site_valid"][:, index].bool().reshape(-1, 1) & complete
        return pooled.flatten(1), target, valid
    denominator = footprint.sum(dim=(-2, -1))
    pooled = (prediction.float() * footprint).sum(dim=(-2, -1)) / denominator.clamp(min=1e-6)
    index = TASKS.index(task)
    target = samples["site_value"][:, index].reshape(-1, 1)
    valid = samples["site_valid"][:, index].bool().reshape(-1, 1) & (denominator > 0)
    if "prediction_valid" in samples:
        valid = valid & ~((footprint > 0) & ~samples["prediction_valid"]).any(dim=(-2, -1))
    return pooled, target, valid


def combined_supervision_values(task, prediction, samples, sm_support=None):
    """Use identical product and optional station support for every head."""
    product = supervision_values(task, prediction, samples, sm_support)
    site = site_supervision_values(task, prediction, samples, sm_support)
    if site is None:
        return product
    return tuple(torch.cat((p.flatten(), s.flatten())) for p, s in zip(product, site))


@dataclass
class PreparedTaskSupervision:
    """Detached support shared by the final and auxiliary heads for one task."""

    task: str
    target: torch.Tensor
    valid: torch.Tensor
    support: torch.Tensor
    count: torch.Tensor
    weights: torch.Tensor | None = None
    denominator: torch.Tensor | None = None
    site_footprint: torch.Tensor | None = None
    site_denominator: torch.Tensor | None = None

    def project(self, prediction):
        prediction = prediction.float()
        if self.task == "water":
            pooled = prediction
        elif self.task == "soil_moisture":
            if self.weights is not None:
                pooled = weighted_sm_mean(prediction, self.weights, self.denominator)
            else:
                pooled = prediction.masked_fill(~self.support, 0).sum(dim=(-2, -1), keepdim=True)
                pooled = pooled / self.denominator
        else:
            pooled = (prediction * self.weights).sum(dim=(-2, -1))
        if self.site_footprint is not None:
            site_prediction = prediction
            if self.task == "soil_moisture" and self.weights is not None:
                site_prediction = prediction.masked_fill(self.site_footprint == 0, 0)
            site = (site_prediction * self.site_footprint).sum(dim=(-2, -1))
            site = site / self.site_denominator
            pooled = torch.cat((pooled.flatten(), site.flatten()))
        return pooled


def prepare_supervision(outputs, samples, sm_support=None):
    """Compute support, completeness and counts once per batch, never across batches."""
    reference = next(iter(outputs.values()))
    support = samples.get("prediction_valid")
    if support is None:
        support = torch.ones_like(reference, dtype=torch.bool)
    footprint = site_denominator = site_complete = None
    if all(key in samples for key in ("site_value", "site_valid", "site_footprint")):
        footprint = samples["site_footprint"].float()
        if footprint.ndim == 3:
            footprint = footprint[:, None]
        site_count = footprint.sum(dim=(-2, -1))
        site_denominator = site_count.clamp(min=1e-6)
        site_complete = (site_count > 0) & ~((footprint > 0) & ~support).any(dim=(-2, -1))
    prepared = {}
    for task, prediction in outputs.items():
        weights = denominator = None
        task_footprint, task_site_denominator = footprint, site_denominator
        task_site_complete = site_complete
        if task == "water":
            target, valid = samples[task], samples[f"{task}_valid"] & support
            shape = prediction.shape
        elif task == "soil_moisture":
            if sm_support is None:
                count = support.sum(dim=(-2, -1), keepdim=True)
                denominator = count.clamp(min=1)
                valid = samples[f"{task}_valid"] & (count > 0)
            else:
                weights, denominator = sm_support.weights, sm_support.denominator
                valid = samples[f"{task}_valid"] & sm_support.valid
                if footprint is not None:
                    task_footprint, task_site_denominator, task_site_complete = (
                        sm_support.station_support(footprint)
                    )
            target, shape = samples[task], denominator.shape
        else:
            if "et_native_weights" not in samples:
                raise ValueError("ET supervision requires native pixel area weights")
            weights = samples["et_native_weights"].float()
            complete = ~((weights > 0) & ~support).any(dim=(-2, -1))
            valid = samples["et_native_valid"] & complete & (weights.sum(dim=(-2, -1)) > 0)
            target, shape = samples["et_native_value"], weights.shape[:-2]
        if footprint is not None:
            index = TASKS.index(task)
            site_valid = samples["site_valid"][:, index].bool().reshape(-1, 1) & task_site_complete
            target = torch.cat((target.flatten(), samples["site_value"][:, index].reshape(-1)))
            valid = torch.cat((valid.flatten(), site_valid.flatten()))
        else:
            shape = torch.broadcast_shapes(shape, target.shape, valid.shape)
            target, valid = target.expand(shape), valid.expand(shape)
        prepared[task] = PreparedTaskSupervision(
            task, target, valid, support, valid.sum(), weights, denominator,
            task_footprint, task_site_denominator,
        )
    return prepared


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
        water_soil_coupling: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        if set(heads) != set(TASKS):
            raise ValueError(f"heads must be exactly {TASKS}, got {tuple(heads)}")
        self.encoder = MODELS.build(encoder)
        self.decoder = MODELS.build(decoder)
        self.heads = torch.nn.ModuleDict({name: MODELS.build(cfg) for name, cfg in heads.items()})
        self.loss_module = MODELS.build(dict(loss, target_statistics=target_statistics))
        self.output_scale = output_scale
        self.water_soil_coupling = (
            WaterSoilCoupling(**water_soil_coupling) if water_soil_coupling is not None else None
        )

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
        return self._loss_from_outputs(outputs, data_samples)

    def _loss_from_outputs(self, outputs, data_samples, auxiliary=None, auxiliary_weights=None):
        coupling = getattr(self, "water_soil_coupling", None)
        sm_support = (
            coupling.prepare(outputs, data_samples, supervision=True)
            if coupling is not None and coupling.enabled else None
        )
        prepared = prepare_supervision(outputs, data_samples, sm_support)
        names, sums, count_indices = [], [], []
        for index, task in enumerate(TASKS):
            plan = prepared[task]
            names.append(f"loss_{task}")
            sums.append(self.loss_module.local_sum(
                task, plan.project(outputs[task]), plan.target, plan.valid
            ))
            count_indices.append(index)
        for task, prediction in (auxiliary or {}).items():
            index = TASKS.index(task)
            plan = prepared[task]
            names.append(f"loss_aux_{task}")
            sums.append(masked_standardized_mse_sum(
                plan.project(prediction), plan.target, plan.valid,
                self.loss_module.target_std[index],
            ))
            count_indices.append(index)
        values, counts = global_observation_means(
            sums, torch.stack([prepared[task].count for task in TASKS]),
            torch.tensor(count_indices, dtype=torch.long, device=sums[0].device),
        )
        objectives = dict(zip(names, values))
        losses: dict[str, torch.Tensor] = {}
        for index, task in enumerate(TASKS):
            losses[f"loss_{task}"] = objectives[f"loss_{task}"]
            losses[f"valid_count_{task}"] = counts[index]
            variance = self.loss_module.variance(task).detach()
            losses[f"variance_{task}"] = variance
            losses[f"precision_{task}"] = variance.reciprocal()
        for task in auxiliary or {}:
            losses[f"loss_aux_{task}"] = auxiliary_weights[task] * objectives[f"loss_aux_{task}"]
        return losses

    def predict(
        self, inputs: dict[str, torch.Tensor], data_samples: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        outputs = self._forward(inputs)
        coupling = getattr(self, "water_soil_coupling", None)
        sm_prediction_support = sm_supervision_support = None
        if coupling is not None and coupling.enabled:
            # Export validity never uses WF labels, including during validation.
            sm_prediction_support = coupling.prepare(outputs, data_samples)
            if data_samples is not None:
                sm_supervision_support = coupling.prepare(outputs, data_samples, supervision=True)
        batch_size = next(iter(outputs.values())).shape[0]
        predictions: list[dict[str, Any]] = []
        metainfo = {} if data_samples is None else data_samples.get("metainfo", {})
        supervised = (
            {}
            if data_samples is None
            else {
                task: supervision_values(task, outputs[task], data_samples, sm_supervision_support)
                for task in TASKS
            }
        )
        for index in range(batch_size):
            item = {task: outputs[task][index] for task in TASKS}
            if sm_prediction_support is not None:
                sm_valid = sm_prediction_support.pixel_valid[index]
                item["soil_moisture"] = item["soil_moisture"].masked_fill(~sm_valid, float("nan"))
                item["soil_moisture_prediction_valid"] = sm_valid
                item["soil_moisture_land_fraction"] = (
                    sm_prediction_support.land_fraction[index].masked_fill(
                        ~sm_prediction_support.known[index], float("nan")
                    )
                )
                item["soil_moisture_supervision_source"] = coupling.supervision_source
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
