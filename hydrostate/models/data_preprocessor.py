"""MMEngine data preprocessor for already aligned Zarr samples."""

from __future__ import annotations

from typing import Any

import torch
from mmengine.model import BaseDataPreprocessor

from hydrostate.registry import MODELS


def aggregate_targets(targets: dict[str, Any], factor: int = 3) -> None:
    """Aggregate water labels; preserve native ET weights and scalar SMAP labels."""
    import torch.nn.functional as F

    water = targets["water"]
    if water.shape[-1] % factor or water.shape[-2] % factor:
        raise ValueError("Label dimensions must be divisible by aggregation factor")
    targets["water"] = F.avg_pool2d(water.float(), factor)
    targets["water_valid"] = F.avg_pool2d(targets["water_valid"].float(), factor).eq(1)
    # Legacy scalar selection is allowed for SMAP, never for native ET weights.
    sm = targets["soil_moisture"]
    if sm.shape[-2:] != (1, 1):
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

    def forward(self, data: dict[str, Any], training: bool = False) -> dict[str, Any]:
        data = self.cast_data(data)
        for key in ("s1", "s2", "landsat"):
            data["inputs"][key] = data["inputs"][key].float()
        data["inputs"]["sensor_valid"] = data["inputs"]["sensor_valid"].bool()
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
            observed.float(), 3
        ).eq(1)
        aggregate_targets(data["data_samples"])
        return data
