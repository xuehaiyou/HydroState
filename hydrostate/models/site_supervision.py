"""Separate station objectives on existing land-SM and derived total-ET fields."""

import json
import math
from pathlib import Path

import torch
from torch import nn

from hydrostate.data.site.contract import OBSERVATIONS
from hydrostate.losses.uncertainty_nll import masked_standardized_mse


def station_values(outputs, samples):
    """No LF scaling of point SM; ET fractions retain gradients through the mixture."""
    if not all(f"{name}_native_value" in samples for name in OBSERVATIONS):
        return {}
    fields = dict(site_sm=outputs["soil_moisture"].float())
    fields["site_et"] = sum(
        outputs[cover].float() * outputs[et].float()
        for cover, et in (("water", "et_water"), ("snow_ice", "et_ice"), ("land", "et_land"))
    )
    result = {}
    for name, prediction in fields.items():
        weights = samples[f"{name}_native_weights"].detach().float()
        support = samples.get("prediction_valid", torch.ones_like(prediction, dtype=torch.bool))
        # Each accepted operator already integrates to one. Invalid/padded rows integrate to zero.
        complete = ~((weights > 0) & ~support).any(dim=(-2, -1))
        total = weights.sum((-2, -1))
        valid = samples[f"{name}_native_valid"].bool() & complete & ((total - 1).abs() < 1e-5)
        pooled = (prediction.masked_fill(weights == 0, 0) * weights).sum((-2, -1))
        result[name] = (pooled, samples[f"{name}_native_value"], valid)
    return result


def station_losses(outputs, samples, statistics, weights):
    result = {}
    for name, (prediction, target, valid) in station_values(outputs, samples).items():
        if name not in statistics:
            continue
        loss, count = masked_standardized_mse(prediction, target, valid, statistics[name]["std"])
        result[f"loss_{name}"] = loss * weights.get(name, 1.0)
        result[f"valid_count_{name}"] = count
    return result


class SiteObjective(nn.Module):
    """An opt-in observation loss; statistics must come from accepted station training rows."""

    def __init__(self, statistics, loss_weights=None):
        super().__init__()
        report = (
            json.loads(Path(statistics).read_text()) if isinstance(statistics, str) else statistics
        )
        if (
            report.get("version") != "site_training_statistics_v1"
            or report.get("split") != "train"
            or report.get("ddof") != 0
            or not report.get("observations")
            or not set(report["observations"]).issubset(OBSERVATIONS)
        ):
            raise ValueError("Station loss requires frozen training-only observation statistics")
        digest = report.get("training_fingerprint", "")
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Station training fingerprint must be SHA256")
        self.names = tuple(report["observations"])
        self.statistics = report
        self.loss_weights = {
            name: float((loss_weights or {}).get(name, 1.0)) for name in self.names
        }
        for name, values in report["observations"].items():
            if (
                values["count"] < 2
                or not math.isfinite(values["std"])
                or values["std"] <= 0
                or not math.isfinite(values["mean"])
            ):
                raise ValueError(f"Invalid station statistics: {name}")
        if any(not math.isfinite(v) or v < 0 for v in self.loss_weights.values()):
            raise ValueError("Station loss weights must be finite and nonnegative")
        self.register_buffer(
            "training_fingerprint", torch.tensor(list(bytes.fromhex(digest)), dtype=torch.uint8)
        )
        self.register_buffer(
            "target_std", torch.tensor([report["observations"][name]["std"] for name in self.names])
        )

    def forward(self, outputs, samples):
        stats = {name: dict(std=self.target_std[i]) for i, name in enumerate(self.names)}
        return station_losses(outputs, samples, stats, self.loss_weights)
