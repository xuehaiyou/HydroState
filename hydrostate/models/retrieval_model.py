"""Architecture-independent behavior for current product-supervised retrieval."""

from dataclasses import replace

import torch
from torch.nn import functional as F

from .water_soil_coupling import WaterSoilCoupling


class AlignedLandCoupling(WaterSoilCoupling):
    """Use observed land only at matching label times; otherwise use window support."""

    def prepare(self, outputs, samples=None, *, supervision=False):
        result = super().prepare(outputs, samples, supervision=supervision)
        if "sm_land_time_aligned" not in (samples or {}):
            raise ValueError("Temporal land coupling requires explicit query alignment")
        if supervision and self.supervision_source != "observed":
            return result
        aligned = samples["sm_land_time_aligned"][:, None, None, None]
        pixel = samples.get(
            "prediction_valid", torch.ones_like(outputs["soil_moisture"], dtype=torch.bool)
        )
        weight = pixel.float()
        total = weight.sum(dim=(-2, -1), keepdim=True)
        return replace(
            result,
            land_fraction=torch.where(aligned, result.land_fraction, weight),
            pixel_valid=torch.where(aligned, result.pixel_valid, pixel),
            known=torch.where(aligned, result.known, pixel),
            weights=torch.where(aligned, result.weights, weight),
            denominator=torch.where(aligned, result.denominator, total.clamp(min=1)),
            valid=torch.where(aligned, result.valid, total > 0),
        )


class RetrievalModelMixin:
    """Shared current-contract SM support and ET export, independent of architecture."""

    def __init__(self, land_time_policy="aligned_or_window", site_supervision=None, **kwargs):
        if land_time_policy not in ("aligned_or_window", "require_aligned"):
            raise ValueError("Unknown land_time_policy")
        self.land_time_policy = land_time_policy
        super().__init__(**kwargs)
        if self.water_soil_coupling is not None:
            self.water_soil_coupling = AlignedLandCoupling(**vars(self.water_soil_coupling))
        if site_supervision is not None:
            from .site_supervision import SiteObjective

            self.site_objective = SiteObjective(**site_supervision)

    def _loss_from_outputs(self, outputs, data_samples, auxiliary=None, auxiliary_weights=None):
        losses = super()._loss_from_outputs(outputs, data_samples, auxiliary, auxiliary_weights)
        if hasattr(self, "site_objective"):
            losses.update(self.site_objective(outputs, data_samples))
        return losses

    def _land_samples(self, inputs, samples):
        coupling = self.water_soil_coupling
        if coupling is None or not coupling.enabled:
            return samples
        aligned = (inputs["target_time"]["water"] == inputs["target_time"]["soil_moisture"]).all(
            dim=-1
        )
        if coupling.ice_task is not None:
            aligned &= (inputs["target_time"][coupling.ice_task]
                        == inputs["target_time"]["soil_moisture"]).all(dim=-1)
        if self.land_time_policy == "require_aligned":
            torch._assert_async(
                aligned.all(), "Observed land constraint requires matching WF/SM query times"
            )
        return dict(samples, sm_land_time_aligned=aligned)

    def loss(self, inputs, data_samples):
        return super().loss(inputs, self._land_samples(inputs, data_samples))

    def predict(self, inputs, data_samples=None):
        if data_samples is None:
            # Inference without labels also must not mask SM with WF at another time.
            values = self._forward(inputs)
            observed = torch.stack([
                inputs[f"{key}_valid"].bool().all(dim=2, keepdim=True).any(dim=1)
                for key in self.encoder.input_keys
            ]).any(dim=0)
            prediction_valid = F.avg_pool2d(observed.float(), self.output_scale).eq(1)
            samples = self._land_samples(inputs, dict(prediction_valid=prediction_valid))
            coupling = self.water_soil_coupling
            support = (
                coupling.prepare(values, samples)
                if coupling is not None and coupling.enabled
                else None
            )
            outputs = []
            for i in range(next(iter(values.values())).shape[0]):
                item = {task: value[i] for task, value in values.items()}
                item["metainfo"] = {}
                item["prediction_valid"] = prediction_valid[i]
                item["target_time"] = {
                    task: bounds[i] for task, bounds in inputs["target_time"].items()
                }
                if support is not None:
                    item["soil_moisture"] = item["soil_moisture"].masked_fill(
                        ~support.pixel_valid[i], float("nan")
                    )
                    item["soil_moisture_prediction_valid"] = support.pixel_valid[i]
                    item["sm_land_time_aligned"] = samples["sm_land_time_aligned"][i]
                outputs.append(item)
            return self._derived_outputs(outputs)
        samples = self._land_samples(inputs, data_samples) if data_samples is not None else None
        outputs = super().predict(inputs, samples)
        if samples is not None and "sm_land_time_aligned" in samples:
            for i, output in enumerate(outputs):
                output["sm_land_time_aligned"] = samples["sm_land_time_aligned"][i]
        for i, output in enumerate(outputs):
            output["target_time"] = {
                task: bounds[i] for task, bounds in inputs["target_time"].items()
            }
        return self._derived_outputs(outputs)

    @staticmethod
    def _derived_outputs(outputs):
        for item in outputs:
            if all(name in item for name in ("et_water", "et_ice", "et_land", "snow_ice", "land")):
                total = (item["water"] * item["et_water"] + item["snow_ice"] * item["et_ice"]
                         + item["land"] * item["et_land"])
                valid = item.get("prediction_valid", torch.ones_like(total, dtype=torch.bool))
                item["et_total"] = total.masked_fill(~valid, float("nan"))
                item["et_total_prediction_valid"] = valid
        return outputs
