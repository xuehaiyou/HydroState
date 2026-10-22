"""Batch native-grid labels with different source-pixel counts."""

import torch
from mmengine.registry import FUNCTIONS
from torch.utils.data import default_collate


@FUNCTIONS.register_module()
def hydro_collate(batch):
    """Pad the varying number of complete source pixels independently per task."""
    native = [
        key.removesuffix("_value")
        for key in batch[0]["data_samples"]
        if key.endswith("_native_value")
    ]
    for prefix in native:
        count = max(item["data_samples"][f"{prefix}_value"].shape[0] for item in batch)
        for item in batch:
            targets = item["data_samples"]
            keys = [f"{prefix}_{suffix}" for suffix in ("weights", "value", "valid")]
            if f"{prefix}_uncertainty" in targets:
                keys.append(f"{prefix}_uncertainty")
            for key in keys:
                value = targets[key]
                padding = value.new_zeros((count - value.shape[0], *value.shape[1:]))
                targets[key] = torch.cat((value, padding))
    return default_collate(batch)
