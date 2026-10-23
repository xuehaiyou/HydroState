"""One satellite input contract for sample readers and regional inference."""

import torch

from .input_products import L2_PROFILE, LEGACY_PROFILE
from .retrieval.contract import PROFILE


def input_keys(profile):
    if profile not in {LEGACY_PROFILE, L2_PROFILE, PROFILE}:
        raise ValueError(f"Unknown model input profile: {profile}")
    if profile == PROFILE:
        from .retrieval.specs import INPUT_SPECS
        return tuple(INPUT_SPECS)
    return ("s1", "s2", "landsat_l2" if profile == L2_PROFILE else "landsat")


def prepare_model_inputs(inputs, profile):
    """Reconcile L2 validity and acquisition times before temporal indexing.

    Accept TCHW samples or BTCHW batches. L2 replaces the third satellite;
    missing L2 observations never fall back to L1 or retain its availability.
    """
    input_keys(profile)
    inputs = dict(inputs)
    if profile in {LEGACY_PROFILE, PROFILE}:
        if "landsat_l2" in inputs:
            raise ValueError("landsat_l2 requires the explicit L2 input profile")
        return inputs
    if "landsat" in inputs or "landsat_valid" in inputs:
        raise ValueError("L2 inputs must replace, not accompany, legacy Landsat L1")
    values, valid = inputs["landsat_l2"], inputs["landsat_l2_valid"].bool()
    times = inputs["landsat_l2_time"]
    if (values.ndim not in (4, 5) or values.shape[-3] != 7 or valid.shape != values.shape
            or times.shape != values.shape[:-3] or times.is_floating_point()):
        raise ValueError("L2 requires T/BT x 7 x H x W values/masks and integer T/BT times")
    if values.shape[:-3] != inputs["s1"].shape[:-3] or (
            values.shape[-2:] != inputs["s1"].shape[-2:]):
        raise ValueError("L2 must share the satellite slots and spatial grid")
    if not torch.isfinite(values.masked_fill(~valid, 0)).all():
        raise ValueError("Nonfinite L2 reflectance marked valid")
    # All seven bands describe a single usable optical observation at each pixel.
    pixels = valid.all(dim=-3, keepdim=True)
    available = pixels.flatten(-3).any(-1)
    if ((times <= 0) & available).any():
        raise ValueError("Valid L2 observations require positive acquisition times")
    sensor = inputs["sensor_valid"].clone().bool()
    stamps = inputs["observation_time"].clone().long()
    if sensor.shape != (*times.shape, 3) or stamps.shape != sensor.shape:
        raise ValueError("Satellite metadata must have T/BT x 3 shape")
    sensor[..., 2] = available
    stamps[..., 2] = times.long().masked_fill(~available, 0)
    inputs.update(landsat_l2=values.float().masked_fill(~pixels, 0),
                  landsat_l2_valid=pixels.expand_as(valid),
                  landsat_l2_time=stamps[..., 2].clone(),
                  sensor_valid=sensor, observation_time=stamps)
    return inputs
