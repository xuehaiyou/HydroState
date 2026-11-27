"""Attach coincident specialist observations before shared spatial augmentation."""

from datetime import UTC, datetime

import numpy as np
import torch

from hydrostate.data.retrieval.contract import INPUT_GROUPS
from hydrostate.data.retrieval.specialist_hls import sidecar


def specialist_inputs(inputs, sample_id, sources, root):
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
                if group == "s1_10m" and "s1" in sources
                else torch.zeros_like(inputs[key])
            )
    if "hls" in sources:
        with np.load(sidecar(root, sample_id), allow_pickle=False) as data:

            def expand(name):
                return (
                    torch.from_numpy(data[name]).repeat_interleave(3, -2).repeat_interleave(3, -1)
                )

            q = expand("quality")
            result["hls"] = expand("hls")
            result["hls_valid"] = q[:, :1] > 0
            result["location"] = torch.from_numpy(data["location"])
        # One joint six-band optical quality record per slot on the aligned decoder grid.
        result.update(s2_10m_quality=q, s2_10m_valid=result["hls_valid"])
        dates = [
            datetime.fromtimestamp(v, UTC) for v in (inputs["slot_bounds"].sum(-1) // 2).tolist()
        ]
        result["calendar"] = torch.tensor([[d.year, d.timetuple().tm_yday] for d in dates])
    return result
