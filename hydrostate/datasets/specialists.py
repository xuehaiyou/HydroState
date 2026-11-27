"""Attach coincident specialist observations before shared spatial augmentation."""

from datetime import UTC, datetime
from functools import lru_cache

import torch
from torch.nn import functional as F

from hydrostate.data.retrieval.contract import INPUT_GROUPS
from hydrostate.data.retrieval.specs import INPUT_SPECS

PRITHVI_BANDS = ("B2", "B3", "B4", "B8A", "B11", "B12")


@lru_cache(maxsize=1)
def s2_encoding():
    from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.data.normalize import load_computed_config

    spec = INPUT_SPECS["s2"]
    stats = load_computed_config()[spec.modality]
    names = [band.name for band in spec.bands]
    indices = [names.index(name) for name in PRITHVI_BANDS]
    selected = [stats[spec.bands[i].normalization_key] for i in indices]
    return indices, *(
        torch.tensor([x[key] for x in selected]).view(1, 6, 1, 1) for key in ("mean", "std")
    )


def specialist_inputs(inputs, sources, location):
    result = {
        key: inputs[key]
        for key in ("slot_bounds", "target_time", "sensor_valid", "observation_time")
    }
    if "s1" in sources:
        result.update(s1=inputs["s1"], s1_valid=inputs["s1_valid"])
    for group in INPUT_GROUPS:
        for suffix in ("quality", "valid"):
            key = group + "_" + suffix
            result[key] = (
                inputs[key]
                if (group == "s1_10m" and "s1" in sources)
                or (group in ("s2_10m", "s2_20m") and "s2" in sources)
                else torch.zeros_like(inputs[key])
            )
    if "s2" in sources:
        indices, mean, std = s2_encoding()
        # Published Zarr normalized DN. Prithvi statistics also use DN (reflectance x 10000).
        dn = (inputs["s2"][:, indices].float() - 0.5) * (4 * std) + mean
        valid = inputs["s2_valid"].bool()
        # Require a completely observed 30 m cell, then use the existing 120-grid model interface.
        valid30 = F.avg_pool2d(valid.float(), 3).eq(1)
        dn30 = F.avg_pool2d(dn.masked_fill(~valid, 0), 3).masked_fill(~valid30, 0)
        result["s2"] = dn30.repeat_interleave(3, -2).repeat_interleave(3, -1)
        result["s2_valid"] = valid30.repeat_interleave(3, -2).repeat_interleave(3, -1)
        result["location"] = torch.as_tensor(location, dtype=torch.float32)
        dates = [
            datetime.fromtimestamp(v, UTC) for v in (inputs["slot_bounds"].sum(-1) // 2).tolist()
        ]
        result["calendar"] = torch.tensor([[d.year, d.timetuple().tm_yday] for d in dates])
    return result
