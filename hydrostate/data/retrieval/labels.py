"""Pure array supervision routing; unknown coverage is never assigned to land."""

import json
from functools import lru_cache

import numpy as np

from hydrostate.data.native_support import NativeSupport

from .contract import COVER, ET, OUTPUT_SCALE, RetrievalSchema
from .specs import LABEL_SPECS


def block_mean(value, factor=OUTPUT_SCALE):
    h, w = value.shape[-2:]
    return value.reshape(*value.shape[:-2], h // factor, factor, w // factor, factor).mean((-3, -1))


def cover_targets(raw, threshold=0.95):
    known = raw["known"].astype(np.float32)
    fraction = block_mean(known)
    targets = dict(
        cover_known=fraction[None],
        cover_days=block_mean(raw["days"])[None],
        cover_time_span=block_mean(raw["last_day"] - raw["first_day"])[None],
        cover_observed_days=block_mean(raw["observed_days"])[None],
        cover_unknown_fraction=block_mean(raw["unknown_fraction"])[None],
    )
    for task in COVER:
        value = block_mean(raw[task] * known) / np.maximum(fraction, 1e-6)
        targets[task] = value[None]
        targets[f"{task}_valid"] = (fraction > threshold)[None].astype(np.uint8)
    # Covers with no observations have all-zero labels and false masks.
    return targets


def sm_support_eligible(values, threshold=0.95, water_ice_limit=0.05):
    known = values.get("known", 0)
    contamination = values.get("water_ice", 1) + (1 - known)
    return known > threshold and contamination < water_ice_limit


def sm_tile_eligible(targets, config):
    tile_land = float((targets["land"] * targets["cover_known"]).mean())
    return tile_land > config["min_tile_land_fraction"]


def ice_centre_eligible(targets, threshold=0.5):
    """Group 7 requires a valid target-window centre, not merely a snowy scene."""
    h, w = targets["snow_ice"].shape[-2:]
    return bool(
        targets["snow_ice_valid"][0, h // 2, w // 2]
        and targets["snow_ice"][0, h // 2, w // 2] >= threshold
    )


def sm_eligible(quality, threshold=0.95, water_ice_limit=0.05):
    """Both 9 km grid and approximate wider footprint must pass independently."""
    for support in ("grid", "footprint"):
        if not sm_support_eligible(quality.get(support, {}), threshold, water_ice_limit):
            return False
    return quality.get("valid_days", 0) > 0 and np.isfinite(quality.get("value", np.nan))


@lru_cache(maxsize=64)
def native_geometry(identity):
    """Reuse geometry across dates/products on the same grids, bounded in memory."""
    sample, projection = json.loads(identity)
    weights, centres = NativeSupport(sample).geometry(projection)
    fine, _ = NativeSupport(sample, factor=1).geometry(projection)
    return weights, centres, fine


def build_targets(sample, cover, pml, sm, config):
    schema = RetrievalSchema()
    targets = {name: np.zeros(spec.tail_shape, spec.dtype) for name, spec in schema.targets.items()}
    targets.update(cover_targets(cover, config["min_known_fraction"]))
    land_known = targets["land_valid"].astype(bool)
    targets["soil_moisture_domain_valid"] = land_known.astype(np.uint8)
    eligible = sm_eligible(sm, config["min_known_fraction"], config["max_water_ice_fraction"])
    eligible &= sm_tile_eligible(targets, config)
    targets["soil_moisture_valid"][:] = eligible
    targets["soil_moisture"][:] = sm.get("value", 0) if eligible else 0
    # Heterogeneous cover/terrain downweights only SM, never the whole tile.
    targets["soil_moisture_weight"][:] = sm.get("representativeness", 1.0) if eligible else 0

    if pml is None:
        return targets
    # Shared geometry at 10 m preserves purity near native boundaries; output
    # weights still integrate the 30 m predictions over the *complete* cell.
    spatial = {key: sample[key] for key in ("crs", "transform", "lon", "lat")}
    weights, centres, fine_weights = native_geometry(
        json.dumps([spatial, pml["projection"]], sort_keys=True)
    )
    if len(weights) > schema.native_capacity:
        raise ValueError("PML support exceeds schema capacity")
    known = cover["known"] > 0
    for task in ET:
        spec = LABEL_SPECS[task]
        cls, band = spec.coverage_class, spec.band
        purity = (fine_weights * (cover[cls] * known)[None]).sum((-2, -1))
        n = len(centres)
        targets[f"{task}_native_weights"][:n] = weights
        for i, (row, col) in enumerate(centres):
            value = pml[band][row, col]
            valid = (
                pml[f"{band}_valid"][row, col]
                and np.isfinite(value)
                and value >= 0
                and purity[i] > config["min_known_fraction"]
            )
            targets[f"{task}_native_valid"][i] = valid
            targets[f"{task}_native_value"][i] = value if valid else 0
    return targets
