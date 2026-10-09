"""Land-conditional SM support, independent of the learned feature exchange."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


def weighted_sm_mean(prediction, weights, denominator):
    """Reduce in FP32, clearing excluded values before multiplication (NaN * 0)."""
    values = prediction.float().masked_fill(weights == 0, 0)
    return (values * weights).sum(dim=(-2, -1), keepdim=True) / denominator


@dataclass
class SoilMoistureSupport:
    land_fraction: torch.Tensor
    pixel_valid: torch.Tensor
    weights: torch.Tensor
    denominator: torch.Tensor
    valid: torch.Tensor
    known: torch.Tensor
    min_window_land_fraction: float

    def project(self, prediction):
        return weighted_sm_mean(prediction, self.weights, self.denominator)

    def station_support(self, footprint):
        """Require known support over the whole station footprint, then weight land."""
        weights = footprint * self.weights
        total = weights.sum(dim=(-2, -1))
        area = footprint.sum(dim=(-2, -1))
        complete = ~((footprint > 0) & ~self.known).any(dim=(-2, -1))
        valid = complete & (total > 0) & (total >= self.min_window_land_fraction * area)
        return weights, total.masked_fill(total == 0, 1), valid


@dataclass(frozen=True)
class WaterSoilCoupling:
    """Observed WF defines supervision; prediction/export always uses predicted WF.

    Both sources are detached. Unknown WF never silently becomes land. The
    existing SMAP window-center approximation is unchanged by this operator.
    Thresholds are configurable engineering cutoffs, not validated physical limits.
    """

    enabled: bool = True
    supervision_source: str = "observed"
    min_land_fraction: float = 0.01
    min_window_land_fraction: float = 0.01

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise ValueError("water_soil_coupling.enabled must be boolean")
        if self.supervision_source not in ("observed", "predicted"):
            raise ValueError("SM supervision_source must be observed or predicted")
        for name in ("min_land_fraction", "min_window_land_fraction"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value < 1:
                raise ValueError(f"{name} must be finite and in [0, 1)")

    def prepare(self, outputs, samples=None, *, supervision=False):
        samples = {} if samples is None else samples
        reference = outputs["soil_moisture"]
        observed = supervision and self.supervision_source == "observed"
        if observed and not all(key in samples for key in ("water", "water_valid")):
            raise ValueError("Observed WF-SM supervision requires water and water_valid")
        water = (samples["water"] if observed else outputs["water"]).detach().float()
        support = samples.get("prediction_valid", torch.ones_like(reference, dtype=torch.bool))
        if water.shape != reference.shape or support.shape != reference.shape:
            raise ValueError("WF, SM and prediction_valid must share the output grid")
        source_valid = torch.isfinite(water) & (water >= 0) & (water <= 1)
        if observed:
            if samples["water_valid"].shape != reference.shape:
                raise ValueError("water_valid must share the output grid")
            source_valid = source_valid & samples["water_valid"].bool()
        known = source_valid & support.bool()
        # No new land-cover data is required. A future trusted domain mask can
        # exclude non-soil surfaces; without it, non-water is only a soil proxy.
        domain = samples.get("soil_moisture_domain_valid", torch.ones_like(known))
        if domain.shape != reference.shape:
            raise ValueError("soil_moisture_domain_valid must share the output grid")
        land_fraction = (1 - water).masked_fill(~known, 0)
        pixel_valid = known & domain.bool() & (land_fraction > self.min_land_fraction)
        weights = land_fraction.masked_fill(~pixel_valid, 0)
        total = weights.sum(dim=(-2, -1), keepdim=True)
        area = support.sum(dim=(-2, -1), keepdim=True)
        complete = ~(support.bool() & ~source_valid).any(dim=(-2, -1), keepdim=True)
        valid = complete & (total > 0) & (total >= self.min_window_land_fraction * area)
        return SoilMoistureSupport(
            land_fraction, pixel_valid, weights, total.masked_fill(total == 0, 1), valid, known,
            self.min_window_land_fraction,
        )
