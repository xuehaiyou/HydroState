"""Validate frozen single/cohort station scales for training and evaluation."""

import json
import math
from pathlib import Path

from hydrostate.data.gee.contracts import fingerprint

from .contract import OBSERVATIONS


def load_site_statistics(source):
    report = json.loads(Path(source).read_text()) if isinstance(source, (str, Path)) else source
    if report.get("split") != "train" or report.get("ddof") != 0:
        raise ValueError("Station statistics must be training-only population moments")
    observations = report.get("observations", {})
    if not observations or not set(observations).issubset(OBSERVATIONS):
        raise ValueError("Missing/unknown station observation statistics")
    for value in observations.values():
        if (
            value["count"] < 2
            or not math.isfinite(value["mean"])
            or not math.isfinite(value["std"])
            or value["std"] <= 0
        ):
            raise ValueError("Invalid station moments")
    version = report.get("version")
    if version == "site_training_statistics_v1":
        digest = report.get("training_fingerprint", "")
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Station training fingerprint must be SHA256")
    elif version == "site_training_statistics_v2":
        parts = report.get("cohorts", [])
        if not parts or any(p.get("version") != "site_training_statistics_v1" for p in parts):
            raise ValueError("Expected explicit single-cohort statistics")
        for part in parts:
            load_site_statistics(part)
        if set(observations) != set().union(*(p["observations"] for p in parts)):
            raise ValueError("Cohort statistics tasks disagree")
        for name, value in observations.items():
            values = [p["observations"][name] for p in parts if name in p["observations"]]
            count = sum(v["count"] for v in values)
            mean = sum(v["count"] * v["mean"] for v in values) / count
            variance = (
                sum(v["count"] * (v["std"] ** 2 + (v["mean"] - mean) ** 2) for v in values) / count
            )
            if (
                value["count"] != count
                or not math.isclose(value["mean"], mean, abs_tol=1e-12)
                or not math.isclose(value["std"], math.sqrt(variance), rel_tol=1e-8)
            ):
                raise ValueError("Combined station moments disagree with cohorts")
        digest = fingerprint(report)
    else:
        raise ValueError("Unsupported station statistics version")
    return report, digest
