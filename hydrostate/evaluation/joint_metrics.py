"""Separate product and station metrics with a fixed validation selection score."""

import math
from collections import defaultdict

from mmengine.evaluator import BaseMetric

from hydrostate.data.site.training_statistics import load_site_statistics
from hydrostate.registry import METRICS

from .hydro_metrics import HydroStateMetric


class SiteMetric(HydroStateMetric):
    def __init__(self, statistics):
        BaseMetric.__init__(self)
        report, _ = load_site_statistics(statistics)
        self.tasks = tuple(report["observations"])
        self.task_scales = {name: report["observations"][name]["std"] for name in self.tasks}
        self.sample_metrics_path = None


@METRICS.register_module()
class JointHydroStateMetric(BaseMetric):
    default_prefix = None

    def __init__(self, target_statistics, site_statistics, tasks, site_score_weight=2 / 3,
                 comparison_years=None, **kwargs):
        super().__init__(**kwargs)
        if not 0 < site_score_weight < 1:
            raise ValueError("Both observation sources must contribute to model selection")
        self.site_score_weight = site_score_weight
        if comparison_years is not None and (
            len(comparison_years) != 2 or comparison_years[0] > comparison_years[1]
        ):
            raise ValueError("comparison_years must be an inclusive start/end pair")
        self.comparison_years = comparison_years
        self.product = HydroStateMetric(target_statistics, tasks)
        self.site = SiteMetric(site_statistics)

    def process(self, data_batch, data_samples):
        for sample in data_samples:
            source = sample["metainfo"]["supervision_source"]
            if source not in ("product", "site"):
                raise ValueError("Missing joint supervision source")
            metric = self.product if source == "product" else self.site
            metric.process(data_batch, [sample])
            result = metric.results.pop()
            result["source"] = source
            if self.comparison_years is not None:
                result["year"] = int(sample["metainfo"]["year"])
            if source == "site":
                result["spatial_group_id"] = sample["metainfo"]["spatial_group_id"]
            self.results.append(result)

    def compute_metrics(self, results):
        output = {}
        scores = {}
        for source, metric in (("product", self.product), ("site", self.site)):
            rows = [r for r in results if r["source"] == source]
            if not rows:
                raise ValueError("Joint evaluation requires both sources")
            values = metric.compute_metrics(rows)
            output.update({f"{source}/{key}": value for key, value in values.items()})
            scores[source] = values["hydro_score"]
            if source == "site":
                output.update(self._group_metrics(rows))
                for frequency in sorted({r["frequency_bin"] for r in rows}):
                    subset = [r for r in rows if r["frequency_bin"] == frequency]
                    for task in metric.tasks:
                        valid = [r for r in subset if r.get(f"{task}/count", 0) > 0]
                        output[f"site/frequency_{frequency}/{task}_spatial_groups"] = len(
                            {r["spatial_group_id"] for r in valid}
                        )
            if self.comparison_years is not None:
                first, last = self.comparison_years
                common = [row for row in rows if first <= row["year"] <= last]
                if common:
                    output.update({f"{source}/years_{first}_{last}/{key}": value
                                   for key, value in metric.compute_metrics(common).items()})
        weight = self.site_score_weight
        output["joint_score"] = (1 - weight) * scores["product"] + weight * scores["site"]
        return output

    def _group_metrics(self, rows):
        """Supplement pooled observations with equal-group summaries, without reselecting models."""
        groups = defaultdict(list)
        for row in rows:
            groups[row["spatial_group_id"]].append(row)
        values = [self.site.compute_metrics(group) for group in groups.values()]
        output = {}
        for task in self.site.tasks:
            eligible = [group for group in values if group.get(f"{task}_count", 0) > 0]
            output[f"site/group_macro/{task}_groups"] = len(eligible)
            for name in ("rmse", "mae", "bias", "r2"):
                numbers = [group[f"{task}_{name}"] for group in eligible
                           if math.isfinite(group[f"{task}_{name}"])]
                if numbers:
                    output[f"site/group_macro/{task}_{name}"] = sum(numbers) / len(numbers)
                    output[f"site/group_macro/{task}_{name}_groups"] = len(numbers)
        return output
