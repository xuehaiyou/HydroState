"""Streaming regression metrics for all three HydroState outputs."""

from __future__ import annotations

import math

import torch
from mmengine.evaluator import BaseMetric

from hydrostate.models.hydrostate_model import TASKS
from hydrostate.registry import METRICS


@METRICS.register_module()
class HydroStateMetric(BaseMetric):
    default_prefix = None

    def __init__(self, task_scales=None, **kwargs):
        super().__init__(**kwargs)
        self.task_scales = task_scales or dict(water=1.0, soil_moisture=0.1, et=3.0)

    def process(self, data_batch, data_samples: list[dict]) -> None:
        for sample in data_samples:
            result: dict[str, float] = {}
            for task in TASKS:
                prediction = sample[f"supervised_{task}"].detach().double()
                target = sample[f"target_{task}"].detach().double()
                valid = sample[f"valid_{task}"].detach().bool()
                valid = valid.expand_as(target)
                if not torch.any(valid):
                    continue
                error = prediction.masked_select(valid) - target.masked_select(valid)
                values = target.masked_select(valid)
                result[f"{task}/count"] = float(error.numel())
                result[f"{task}/sum_abs"] = float(error.abs().sum())
                result[f"{task}/sum_sq"] = float((error**2).sum())
                result[f"{task}/sum_err"] = float(error.sum())
                result[f"{task}/sum_y"] = float(values.sum())
                result[f"{task}/sum_y2"] = float((values**2).sum())
            self.results.append(result)

    def compute_metrics(self, results: list[dict]) -> dict[str, float]:
        metrics: dict[str, float] = {}
        rmse_values = []
        for task in TASKS:
            totals = {
                key: sum(item.get(f"{task}/{key}", 0.0) for item in results)
                for key in ("count", "sum_abs", "sum_sq", "sum_err", "sum_y", "sum_y2")
            }
            count = totals["count"]
            if count == 0:
                continue
            mae = totals["sum_abs"] / count
            mse = totals["sum_sq"] / count
            rmse = math.sqrt(mse)
            bias = totals["sum_err"] / count
            total_variance = totals["sum_y2"] - totals["sum_y"] ** 2 / count
            r2 = (
                1.0 - totals["sum_sq"] / total_variance
                if count > 1 and total_variance > 1e-12 * max(totals["sum_y2"], 1.0)
                else float("nan")
            )
            metrics.update(
                {
                    f"{task}_mae": mae,
                    f"{task}_rmse": rmse,
                    f"{task}_bias": bias,
                    f"{task}_r2": r2,
                }
            )
            rmse_values.append(rmse / self.task_scales[task])
        # Normalize physical units with the same configurable scales as the loss.
        metrics["hydro_score"] = -sum(rmse_values) / len(rmse_values) if rmse_values else -math.inf
        return metrics
