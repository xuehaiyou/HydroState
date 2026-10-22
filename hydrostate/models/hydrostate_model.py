"""MMEngine joint HydroState retrieval model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from mmengine.model import BaseModel

from hydrostate.data.input_products import LEGACY_PROFILE
from hydrostate.losses.uncertainty_nll import (
    global_observation_means,
    masked_standardized_mse_sum,
)
from hydrostate.registry import MODELS
from hydrostate.tasks import LEGACY_TASKS, resolve_tasks, task_config

from .water_soil_coupling import WaterSoilCoupling, weighted_sm_mean


def supervision_values(task, prediction, samples, sm_support=None, tasks=LEGACY_TASKS):
    """Return predictions and labels on the same observational support."""
    prediction = prediction.float()  # Stable reductions under BF16 inference/training.
    support = samples.get("prediction_valid", torch.ones_like(prediction, dtype=torch.bool))
    spec = resolve_tasks(tasks)[task]
    if spec.support == "pixel":
        if samples[task].shape != prediction.shape:
            raise ValueError(f"Pixel task {task} must match the model output grid")
        return prediction, samples[task], samples[f"{task}_valid"] & support
    if spec.support in {"window_mean", "tile_mean_weak"}:
        if spec.support == "tile_mean_weak" and sm_support is None:
            raise ValueError("SM weak supervision requires observed land/ice support")
        if task == "soil_moisture" and sm_support is not None:
            return (sm_support.project(prediction), samples[task],
                    samples[f"{task}_valid"] & sm_support.valid)
        count = support.sum(dim=(-2, -1), keepdim=True)
        pooled = prediction.masked_fill(~support, 0).sum(dim=(-2, -1), keepdim=True)
        pooled = pooled / count.clamp(min=1)
        return pooled, samples[task], samples[f"{task}_valid"] & (count > 0)
    if f"{task}_native_weights" not in samples:
        raise ValueError(f"{task} supervision requires native pixel area weights")
    weights = samples[f"{task}_native_weights"].float()
    pooled = (prediction * weights).sum(dim=(-2, -1))
    complete = ~((weights > 0) & ~support).any(dim=(-2, -1))
    valid = samples[f"{task}_native_valid"] & complete & (weights.sum(dim=(-2, -1)) > 0)
    return pooled, samples[f"{task}_native_value"], valid


def site_supervision_values(task, prediction, samples, sm_support=None, tasks=LEGACY_TASKS):
    """Optional station observation on a completely observed, nonempty footprint."""
    index = resolve_tasks(tasks)[task].station_index
    if index is None or not all(
        key in samples for key in ("site_value", "site_valid", "site_footprint")
    ):
        return None
    footprint = samples["site_footprint"].float()
    if footprint.ndim == 3:
        footprint = footprint[:, None]
    if task == "soil_moisture" and sm_support is not None:
        weights, denominator, complete = sm_support.station_support(footprint)
        pooled = weighted_sm_mean(prediction, weights, denominator[..., None, None])
        target = samples["site_value"][:, index].reshape(-1, 1)
        valid = samples["site_valid"][:, index].bool().reshape(-1, 1) & complete
        return pooled.flatten(1), target, valid
    denominator = footprint.sum(dim=(-2, -1))
    pooled = (prediction.float() * footprint).sum(dim=(-2, -1)) / denominator.clamp(min=1e-6)
    target = samples["site_value"][:, index].reshape(-1, 1)
    valid = samples["site_valid"][:, index].bool().reshape(-1, 1) & (denominator > 0)
    if "prediction_valid" in samples:
        valid = valid & ~((footprint > 0) & ~samples["prediction_valid"]).any(dim=(-2, -1))
    return pooled, target, valid


def combined_supervision_values(task, prediction, samples, sm_support=None, tasks=LEGACY_TASKS):
    """Use identical product and optional station support for every head."""
    product = supervision_values(task, prediction, samples, sm_support, tasks)
    site = site_supervision_values(task, prediction, samples, sm_support, tasks)
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
    kind: str | None = None

    def project(self, prediction):
        prediction = prediction.float()
        kind = self.kind or LEGACY_TASKS[self.task].support
        if kind == "pixel":
            pooled = prediction
        elif kind in {"window_mean", "tile_mean_weak"}:
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


def prepare_supervision(outputs, samples, sm_support=None, tasks=LEGACY_TASKS):
    """Compute support, completeness and counts once per batch, never across batches."""
    specs = resolve_tasks(tasks)
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
        spec = specs[task]
        weights = denominator = None
        task_footprint = footprint if spec.station_index is not None else None
        task_site_denominator = site_denominator if task_footprint is not None else None
        task_site_complete = site_complete
        if spec.support == "pixel":
            if samples[task].shape != prediction.shape:
                raise ValueError(f"Pixel task {task} must match the model output grid")
            target, valid = samples[task], samples[f"{task}_valid"] & support
            shape = prediction.shape
        elif spec.support in {"window_mean", "tile_mean_weak"}:
            if spec.support == "tile_mean_weak" and sm_support is None:
                raise ValueError("SM weak supervision requires observed land/ice support")
            if sm_support is None or task != "soil_moisture":
                count = support.sum(dim=(-2, -1), keepdim=True)
                denominator = count.clamp(min=1)
                valid = samples[f"{task}_valid"] & (count > 0)
            else:
                weights, denominator = sm_support.weights, sm_support.denominator
                valid = samples[f"{task}_valid"] & sm_support.valid
                if task_footprint is not None:
                    task_footprint, task_site_denominator, task_site_complete = (
                        sm_support.station_support(footprint)
                    )
            target, shape = samples[task], denominator.shape
        else:
            if f"{task}_native_weights" not in samples:
                raise ValueError(f"{task} supervision requires native pixel area weights")
            weights = samples[f"{task}_native_weights"].float()
            complete = ~((weights > 0) & ~support).any(dim=(-2, -1))
            valid = samples[f"{task}_native_valid"] & complete & (weights.sum(dim=(-2, -1)) > 0)
            target, shape = samples[f"{task}_native_value"], weights.shape[:-2]
        if task_footprint is not None:
            index = spec.station_index
            site_valid = samples["site_valid"][:, index].bool().reshape(-1, 1) & task_site_complete
            target = torch.cat((target.flatten(), samples["site_value"][:, index].reshape(-1)))
            valid = torch.cat((valid.flatten(), site_valid.flatten()))
        else:
            shape = torch.broadcast_shapes(shape, target.shape, valid.shape)
            target, valid = target.expand(shape), valid.expand(shape)
        prepared[task] = PreparedTaskSupervision(
            task, target, valid, support, valid.sum(), weights, denominator,
            task_footprint, task_site_denominator, spec.support,
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
        tasks: dict | None = LEGACY_TASKS,
        input_fusion: dict | None = None,
        fraction_groups: list[list[str]] | None = None,
        task_loss_weights: dict[str, float] | None = None,
    ) -> None:
        self.task_specs = resolve_tasks(tasks)
        self.tasks = tuple(self.task_specs)
        self.task_loss_weights = {name: float((task_loss_weights or {}).get(name, 1.))
                                  for name in self.tasks}
        if any(value < 0 or not torch.isfinite(torch.tensor(value))
               for value in self.task_loss_weights.values()):
            raise ValueError("Task loss weights must be finite and nonnegative")
        self.input_profile = encoder.get("input_profile", LEGACY_PROFILE)
        if data_preprocessor is not None:
            data_preprocessor = dict(data_preprocessor)
            if str(data_preprocessor.get("type", "")).endswith("HydroDataPreprocessor"):
                if (data_preprocessor.get("tasks") is not None
                        and resolve_tasks(data_preprocessor["tasks"]) != self.task_specs):
                    raise ValueError("Preprocessor tasks disagree with model tasks")
                if data_preprocessor.get("output_scale", output_scale) != output_scale:
                    raise ValueError("Preprocessor output_scale disagrees with model")
                data_preprocessor.setdefault("tasks", task_config(self.task_specs))
                data_preprocessor.setdefault("output_scale", output_scale)
                if data_preprocessor.get("input_profile", self.input_profile) != self.input_profile:
                    raise ValueError("Preprocessor input_profile disagrees with encoder")
                data_preprocessor.setdefault("input_profile", self.input_profile)
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        if set(heads) != set(self.tasks):
            raise ValueError(f"heads must be exactly {self.tasks}, got {tuple(heads)}")
        self.fraction_groups = fraction_groups or []
        grouped = set()
        for group in self.fraction_groups:
            if (len(group) < 2 or len(set(group)) != len(group)
                    or grouped.intersection(group) or not set(group).issubset(heads)):
                raise ValueError("Fraction groups require disjoint groups of configured heads")
            if any(self.task_specs[name].support != "pixel"
                   or heads[name].get("activation", "identity") != "identity"
                   or heads[name].get("out_channels", 1) != 1 for name in group):
                raise ValueError("Fraction groups require pixel tasks with identity/logit heads")
            grouped.update(group)
        decoder = dict(decoder)
        if str(decoder.get("type", "")).endswith("StagedTaskDecoder"):
            if (decoder.get("tasks") is not None
                    and resolve_tasks(decoder["tasks"]) != self.task_specs):
                raise ValueError("Decoder tasks disagree with model tasks")
            decoder.setdefault("tasks", task_config(self.task_specs))
        self.encoder = MODELS.build(encoder)
        self.decoder = MODELS.build(decoder)
        self.input_fusion = MODELS.build(input_fusion) if input_fusion else None
        self.heads = torch.nn.ModuleDict({name: MODELS.build(cfg) for name, cfg in heads.items()})
        self.loss_module = MODELS.build(dict(
            loss, target_statistics=target_statistics, tasks=task_config(self.task_specs)))
        if type(output_scale) is not int or output_scale < 1:
            raise ValueError("output_scale must be a positive integer")
        self.output_scale = output_scale
        self.water_soil_coupling = (
            WaterSoilCoupling(**water_soil_coupling) if water_soil_coupling is not None else None
        )
        if self.water_soil_coupling is not None and not {"water", "soil_moisture"}.issubset(heads):
            raise ValueError("Water-soil coupling requires water and soil_moisture tasks")

    def _encode_inputs(self, inputs):
        features = self.encoder(inputs)
        return self.input_fusion(features, inputs) if self.input_fusion is not None else features

    def _forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        features = self._encode_inputs(inputs)
        decoded = self.decoder(features)
        if decoded.shape[-1] % self.output_scale or decoded.shape[-2] % self.output_scale:
            raise ValueError("Decoded grid must be divisible by output_scale")
        decoded = torch.nn.functional.avg_pool2d(decoded, self.output_scale)
        return self._constrain_outputs({name: self.heads[name](decoded) for name in self.tasks})

    def _constrain_outputs(self, outputs):
        """A joint softmax makes exhaustive land-cover fractions sum to one."""
        for group in getattr(self, "fraction_groups", []):
            if not set(group).intersection(outputs):
                continue
            if not set(group).issubset(outputs):
                raise ValueError("Auxiliary supervision must include an entire fraction group")
            fractions = torch.cat([outputs[name] for name in group], dim=1).softmax(dim=1)
            outputs.update(zip(group, fractions.split(1, dim=1)))
        return outputs

    def loss(
        self, inputs: dict[str, torch.Tensor], data_samples: dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        outputs = self._forward(inputs)
        return self._loss_from_outputs(outputs, data_samples)

    def _loss_from_outputs(self, outputs, data_samples, auxiliary=None, auxiliary_weights=None):
        task_specs = getattr(self, "task_specs", LEGACY_TASKS)
        all_tasks = tuple(resolve_tasks(task_specs))
        task_specs = {task: spec for task, spec in resolve_tasks(task_specs).items()
                      if task in outputs}
        tasks = tuple(task_specs)
        coupling = getattr(self, "water_soil_coupling", None)
        sm_support = (
            coupling.prepare(outputs, data_samples, supervision=True)
            if coupling is not None and coupling.enabled else None
        )
        prepared = prepare_supervision(outputs, data_samples, sm_support, task_specs)
        names, sums, count_indices = [], [], []
        for index, task in enumerate(tasks):
            plan = prepared[task]
            names.append(f"loss_{task}")
            sample_weight = data_samples.get(f"{task}_weight")
            options = {} if sample_weight is None else dict(weight=sample_weight)
            sums.append(self.loss_module.local_sum(
                task, plan.project(outputs[task]), plan.target, plan.valid, **options
            ))
            count_indices.append(index)
        for task, prediction in (auxiliary or {}).items():
            index = tasks.index(task)
            plan = prepared[task]
            names.append(f"loss_aux_{task}")
            sums.append(masked_standardized_mse_sum(
                plan.project(prediction), plan.target, plan.valid,
                self.loss_module.target_std[all_tasks.index(task)],
            ))
            count_indices.append(index)
        values, counts = global_observation_means(
            sums, torch.stack([prepared[task].count for task in tasks]),
            torch.tensor(count_indices, dtype=torch.long, device=sums[0].device),
        )
        objectives = dict(zip(names, values))
        losses: dict[str, torch.Tensor] = {}
        for index, task in enumerate(tasks):
            weight = getattr(self, "task_loss_weights", {}).get(task, 1.)
            losses[f"loss_{task}"] = weight * objectives[f"loss_{task}"]
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
        task_specs = getattr(self, "task_specs", LEGACY_TASKS)
        outputs = self._forward(inputs)
        task_specs = {task: spec for task, spec in resolve_tasks(task_specs).items()
                      if task in outputs}
        tasks = tuple(task_specs)
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
                task: supervision_values(
                    task, outputs[task], data_samples, sm_supervision_support, task_specs)
                for task in tasks
            }
        )
        for index in range(batch_size):
            item = {task: outputs[task][index] for task in tasks}
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
                for key in ("prediction_valid", *(f"{task}_native_weights" for task in tasks)):
                    if key in data_samples:
                        item[key] = data_samples[key][index]
                for task in tasks:
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
