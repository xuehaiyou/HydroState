"""Frozen target statistics shared by training, checkpoints and evaluation."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

TASKS = ("water", "soil_moisture", "et")
SUPPORT_VERSION = "water30_smap_center_pml_native_observed_v1"


def training_fingerprint(table) -> str:
    """Hash training records only, independently of manifest row/column order."""
    train = table.loc[table.split == "train"].sort_values("sample_id")
    if train.empty or train.sample_id.duplicated().any():
        raise ValueError("Training manifest must have unique, nonempty training rows")
    content = train.reindex(sorted(train.columns), axis=1).to_json(
        orient="records", date_format="iso", double_precision=15
    )
    return hashlib.sha256(content.encode()).hexdigest()


def load_target_statistics(source: str | dict) -> dict:
    report = (
        json.loads(Path(source).read_text()) if isinstance(source, str) else copy.deepcopy(source)
    )
    if not isinstance(report, dict):
        raise ValueError("Provide a target-statistics JSON path or its complete contents")
    if (
        report.get("version") != 1
        or report.get("split") != "train"
        or report.get("support_version") != SUPPORT_VERSION
        or report.get("ddof") != 0
    ):
        raise ValueError(
            "Target statistics must use training-only, population statistics on v1 support"
        )
    fingerprint = report.get("training_fingerprint", "")
    if len(fingerprint) != 64 or any(c not in "0123456789abcdef" for c in fingerprint):
        raise ValueError("Target statistics require a SHA256 training fingerprint")
    if set(report.get("tasks", {})) != set(TASKS):
        raise ValueError(f"Target statistics must contain exactly {TASKS}")
    for task, values in report["tasks"].items():
        if values.get("count", 0) < 2:
            raise ValueError(f"Too few training labels for {task}")
        if (
            not math.isfinite(values["mean"])
            or not math.isfinite(values["std"])
            or values["std"] <= 0
        ):
            raise ValueError(f"Invalid training mean/std for {task}")
    return report
