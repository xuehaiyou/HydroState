"""MMEngine data preprocessor for already aligned Zarr samples."""

from __future__ import annotations

from typing import Any

import torch
from mmengine.model import BaseDataPreprocessor

from hydrostate.registry import MODELS
from hydrostate.tasks import resolve_tasks

from .temporal_metadata import prepare_temporal_metadata


def aggregate_targets(targets: dict[str, Any], factor: int = 3, tasks=None) -> None:
    """Aggregate water labels; preserve native ET weights and scalar SMAP labels."""
    import torch.nn.functional as F

    for name, spec in resolve_tasks(tasks).items():
        if spec.support != "pixel" or spec.grid != "input":
            continue
        value, valid = targets[name], targets[f"{name}_valid"]
        if value.shape[-1] % factor or value.shape[-2] % factor:
            raise ValueError("Label dimensions must be divisible by aggregation factor")
        targets[name] = F.avg_pool2d(value.float().masked_fill(~valid, 0), factor)
        targets[f"{name}_valid"] = F.avg_pool2d(valid.float(), factor).eq(1)
    # Legacy scalar selection is allowed for SMAP, never for native ET weights.
    sm = targets.get("soil_moisture")
    if sm is not None and sm.shape[-2:] != (1, 1):
        h, w = sm.shape[-2:]
        for key in ("soil_moisture", "soil_moisture_valid"):
            targets[key] = targets[key][..., h // 2 : h // 2 + 1, w // 2 : w // 2 + 1]
    if "site_footprint" in targets:
        targets["site_footprint"] = (
            F.avg_pool2d(targets["site_footprint"].float(), factor) * factor**2
        )


@MODELS.register_module()
class HydroDataPreprocessor(BaseDataPreprocessor):
    """Move nested batches to the target device and enforce model dtypes.

    Satellite normalization is performed while creating Zarr samples. This is
    intentional: the training hot path should not repeat preprocessing, and the
    same preprocessing version is recorded in every shard's attributes.
    """

    def __init__(self, tasks=None, output_scale=3, **kwargs):
        super().__init__(**kwargs)
        if type(output_scale) is not int or output_scale < 1:
            raise ValueError("output_scale must be a positive integer")
        self.tasks = resolve_tasks(tasks)
        self.output_scale = output_scale

    def forward(self, data: dict[str, Any], training: bool = False) -> dict[str, Any]:
        # Keep control metadata on CPU. Pack all gather indices into a single
        # tensor so cast_data transfers them once alongside the image tensors.
        inputs = dict(data["inputs"])
        metadata = None
        temporal = {
            key: inputs.pop(key).detach().to(device="cpu", dtype=dtype)
            for key, dtype in (("sensor_valid", torch.bool), ("observation_time", torch.long))
            if key in inputs
        }
        if "observation_time" in temporal:
            metadata, inputs["_temporal_indices"] = prepare_temporal_metadata(
                temporal["sensor_valid"], temporal["observation_time"]
            )
        data = dict(data, inputs=inputs)
        data = self.cast_data(data)
        data["inputs"].update(temporal)
        if metadata is not None:
            data["inputs"]["_temporal_metadata"] = metadata
        for key in ("s1", "s2", "landsat"):
            data["inputs"][key] = data["inputs"][key].float()
        for key, value in data["data_samples"].items():
            if key.endswith("_valid") and torch.is_tensor(value):
                data["data_samples"][key] = value.bool()
        inputs = data["inputs"]
        masks = [
            inputs[f"{key}_valid"].bool().any(dim=1)
            for key in ("s1", "s2", "landsat")
            if f"{key}_valid" in inputs
        ]
        if masks:
            observed = torch.stack(masks).any(dim=0)
        else:
            raise ValueError("Per-pixel sensor masks are required for spatial supervision")
        data["data_samples"]["prediction_valid"] = torch.nn.functional.avg_pool2d(
            observed.float(), self.output_scale
        ).eq(1)
        aggregate_targets(data["data_samples"], self.output_scale, self.tasks)
        return data
