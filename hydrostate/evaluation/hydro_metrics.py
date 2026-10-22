"""Streaming regression metrics for all three HydroState outputs."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from mmengine.evaluator import BaseMetric

from hydrostate.normalization import load_target_statistics
from hydrostate.registry import METRICS
from hydrostate.tasks import LEGACY_TASKS, resolve_tasks


@METRICS.register_module()
class HydroStateMetric(BaseMetric):
    default_prefix = None

    def __init__(self, target_statistics, tasks=LEGACY_TASKS, evaluated_tasks=None,
                 sample_metrics_path=None, **kwargs):
        super().__init__(**kwargs)
        configured = tuple(resolve_tasks(tasks))
        self.tasks = configured if evaluated_tasks is None else tuple(evaluated_tasks)
        if (not self.tasks or len(set(self.tasks)) != len(self.tasks)
                or not set(self.tasks) <= set(configured)):
            raise ValueError("evaluated_tasks must be unique configured tasks")
        self.sample_metrics_path = sample_metrics_path
        statistics = load_target_statistics(target_statistics, tasks)
        self.task_scales = {task: statistics["tasks"][task]["std"] for task in self.tasks}

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
            for task in self.tasks:
                prediction = sample[f"supervised_{task}"].detach().double()
                target = sample[f"target_{task}"].detach().double()
                valid = sample[f"valid_{task}"].detach().bool()
                valid = valid.expand_as(target)
                if not torch.any(valid):
                    continue
                error = prediction.masked_select(valid) - target.masked_select(valid)
                values = target.masked_select(valid)
                estimates = prediction.masked_select(valid)
                result[f"{task}/count"] = float(error.numel())
                result[f"{task}/sum_abs"] = float(error.abs().sum())
                result[f"{task}/sum_sq"] = float((error**2).sum())
                result[f"{task}/sum_err"] = float(error.sum())
                result[f"{task}/sum_y"] = float(values.sum())
                result[f"{task}/sum_y2"] = float((values**2).sum())
                result[f"{task}/sum_x"] = float(estimates.sum())
                result[f"{task}/sum_x2"] = float((estimates**2).sum())
                result[f"{task}/sum_xy"] = float((estimates * values).sum())
            info = sample.get("metainfo", {})
            if "frequency_bin" in info:
                result["frequency_bin"] = str(info["frequency_bin"])
            if self.sample_metrics_path is not None:
                info = sample.get("metainfo", {})
                result["sample_id"] = str(info["sample_id"])
            self.results.append(result)

    def compute_metrics(self, results: list[dict]) -> dict[str, float]:
        if self.sample_metrics_path is not None:
            destination = Path(self.sample_metrics_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("x") as stream:
                for result in results:
                    stream.write(json.dumps(result, allow_nan=False) + "\n")
        metrics: dict[str, float] = {}
        rmse_values = []
        for task in self.tasks:
            totals = {
                key: sum(item.get(f"{task}/{key}", 0.0) for item in results)
                for key in ("count", "sum_abs", "sum_sq", "sum_err", "sum_y", "sum_y2",
                            "sum_x", "sum_x2", "sum_xy")
            }
            count = totals["count"]
            if count == 0:
                continue
            mae = totals["sum_abs"] / count
            mse = totals["sum_sq"] / count
            rmse = math.sqrt(mse)
            bias = totals["sum_err"] / count
            total_variance = totals["sum_y2"] - totals["sum_y"] ** 2 / count
            prediction_variance = totals["sum_x2"] - totals["sum_x"] ** 2 / count
            covariance = totals["sum_xy"] - totals["sum_x"] * totals["sum_y"] / count
            correlation = (covariance / math.sqrt(total_variance * prediction_variance)
                           if total_variance > 1e-12 and prediction_variance > 1e-12
                           else float("nan"))
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
                    f"{task}_ubrmse": math.sqrt(max(0., mse - bias**2)),
                    f"{task}_correlation": correlation,
                    f"{task}_count": count,
                }
            )
            rmse_values.append(rmse / self.task_scales[task])
        # Keep model selection fixed: use training std, never learned task variance.
        metrics["hydro_score"] = -sum(rmse_values) / len(rmse_values) if rmse_values else -math.inf
        # Bin-specific metrics never change the overall model-selection score.
        for group in sorted({row["frequency_bin"] for row in results if "frequency_bin" in row}):
            rows = [row for row in results if row.get("frequency_bin") == group]
            for task in self.tasks:
                count = sum(row.get(f"{task}/count", 0.) for row in rows)
                metrics[f"frequency_{group}/{task}_count"] = count
                if count:
                    square = sum(row.get(f"{task}/sum_sq", 0.) for row in rows)
                    metrics[f"frequency_{group}/{task}_rmse"] = math.sqrt(square / count)
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
