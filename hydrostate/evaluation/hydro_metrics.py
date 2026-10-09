"""Streaming regression metrics for all three HydroState outputs."""

from __future__ import annotations

import math

import torch
from mmengine.evaluator import BaseMetric

from hydrostate.normalization import TASKS, load_target_statistics
from hydrostate.registry import METRICS


@METRICS.register_module()
class HydroStateMetric(BaseMetric):
    default_prefix = None

    def __init__(self, target_statistics, **kwargs):
        super().__init__(**kwargs)
        statistics = load_target_statistics(target_statistics)
        self.task_scales = {task: statistics["tasks"][task]["std"] for task in TASKS}

    def process(self, data_batch, data_samples: list[dict]) -> None:
        for sample in data_samples:
            result: dict[str, float] = {}
            if "soil_moisture_prediction_valid" in sample:
                support = sample.get("prediction_valid")
                sm_valid = sample["soil_moisture_prediction_valid"].detach().bool()
                if support is None:
                    support = torch.ones_like(sm_valid)
                result["sm_coupling/enabled"] = 1.0
                result["sm_coupling/output_valid"] = float((sm_valid & support).sum())
                result["sm_coupling/input_valid"] = float(support.sum())
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
        # Keep model selection fixed: use training std, never learned task variance.
        metrics["hydro_score"] = -sum(rmse_values) / len(rmse_values) if rmse_values else -math.inf
        if any(item.get("sm_coupling/enabled", 0) for item in results):
            metrics["soil_moisture_observation_count"] = sum(
                item.get("soil_moisture/count", 0.0) for item in results
            )
            observed = sum(item.get("sm_coupling/input_valid", 0.0) for item in results)
            predicted = sum(item.get("sm_coupling/output_valid", 0.0) for item in results)
            metrics["soil_moisture_prediction_coverage"] = (
                predicted / observed if observed else float("nan")
            )
        return metrics
