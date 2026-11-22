"""Separate human validation sources, with fixed Cover/SM/ET model selection."""

import numpy as np

from hydrostate.registry import METRICS

from .joint_metrics import JointHydroStateMetric
from .manual_cover import Scores


@METRICS.register_module()
class ManualJointMetric(JointHydroStateMetric):
    def process(self, data_batch, data_samples):
        for sample in data_samples:
            if sample["metainfo"]["supervision_source"] != "manual":
                super().process(data_batch, [sample])
                continue
            source = sample["metainfo"]["source"]
            score = Scores(source)
            predicted = np.stack(
                [sample[c].detach().cpu().numpy()[0] for c in ("water", "snow_ice", "land")]
            )
            labels = {
                c + suffix: sample["manual_" + c + suffix].detach().cpu().numpy()
                for c in ("water", "snow_ice", "land")
                for suffix in ("", "_valid")
            }
            valid = sample["prediction_valid"].detach().cpu().numpy()[0].astype(bool)
            score.add(predicted, labels, valid)
            self.results.append(
                dict(
                    source="manual",
                    kind=source,
                    region=sample["metainfo"]["region"],
                    sums=score.sums,
                    confusion=score.confusion,
                )
            )

    def compute_metrics(self, results):
        output = super().compute_metrics([r for r in results if r["source"] != "manual"])
        human = [r for r in results if r["source"] == "manual"]
        means = {}
        for kind in ("glacier", "sen1floods11"):
            selected = [r for r in human if r["kind"] == kind]
            if not selected:
                raise ValueError("Both manual validation sources are required for model selection")
            groups = {"all": selected}
            groups.update(
                {
                    region: [r for r in selected if r["region"] == region]
                    for region in sorted({r["region"] for r in selected})
                }
            )
            for group, rows in groups.items():
                score = Scores(kind)
                for row in rows:
                    for name in score.sums:
                        score.sums[name] += row["sums"][name]
                    score.confusion += row["confusion"]
                    score.windows += 1
                report = score.report()
                if report["accuracy"] is not None:
                    output[f"cover/{kind}/{group}/accuracy"] = report["accuracy"]
                for i, actual in enumerate(report["class_order"]):
                    for j, predicted in enumerate(report["class_order"]):
                        key = f"cover/{kind}/{group}/confusion_{actual}_{predicted}"
                        output[key] = int(score.confusion[i, j])
                for name, metrics in report["fraction_metrics"].items():
                    for key, value in metrics.items():
                        if value is not None:
                            output[f"cover/{kind}/{group}/{name}_{key}"] = value
                for name, metrics in report["classes"].items():
                    for key, value in metrics.items():
                        if value is not None:
                            output[f"cover/{kind}/{group}/{name}_{key}"] = value
                if group == "all":
                    values = [x["mae"] for x in report["fraction_metrics"].values()]
                    if any(x is None for x in values):
                        raise ValueError("Missing fraction validation support")
                    means[kind] = sum(values) / len(values)
        cover_error = (means["glacier"] + means["sen1floods11"]) / 2
        station_errors = [
            output[f"site/{t}_rmse"] / self.site.task_scales[t] for t in ("site_sm", "site_et")
        ]
        output["manual_cover_error"] = cover_error
        output["manual_joint_score"] = -(cover_error + sum(station_errors)) / 3
        return output
