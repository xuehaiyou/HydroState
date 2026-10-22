"""Frozen target statistics shared by training, checkpoints and evaluation."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

from hydrostate.tasks import LEGACY_TASKS, resolve_tasks, task_config

LEGACY_TASK_NAMES = tuple(LEGACY_TASKS)
SUPPORT_VERSION = "water30_smap_center_pml_native_observed_v1"
CONFIGURED_SUPPORT_VERSION = "configured_observation_support_v1"


def training_fingerprint(table, data_contract=None) -> str:
    """Hash training records only, independently of manifest row/column order."""
    train = table.loc[table.split == "train"].sort_values("sample_id")
    if train.empty or train.sample_id.duplicated().any():
        raise ValueError("Training manifest must have unique, nonempty training rows")
    content = train.reindex(sorted(train.columns), axis=1).to_json(
        orient="records", date_format="iso", double_precision=15
    )
    if data_contract is not None:
        content += "\n" + json.dumps(data_contract, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(content.encode()).hexdigest()


def load_target_statistics(source: str | dict, tasks=LEGACY_TASKS) -> dict:
    report = (
        json.loads(Path(source).read_text()) if isinstance(source, str) else copy.deepcopy(source)
    )
    if not isinstance(report, dict):
        raise ValueError("Provide a target-statistics JSON path or its complete contents")
    specs = resolve_tasks(tasks)
    version = report.get("version")
    if (
        version not in (1, 2)
        or report.get("split") != "train"
        or report.get("support_version") != (
            SUPPORT_VERSION if version == 1 else CONFIGURED_SUPPORT_VERSION)
        or report.get("ddof") != 0
    ):
        raise ValueError(
            "Target statistics must use training-only, population statistics on declared support"
        )
    fingerprint = report.get("training_fingerprint", "")
    if len(fingerprint) != 64 or any(c not in "0123456789abcdef" for c in fingerprint):
        raise ValueError("Target statistics require a SHA256 training fingerprint")
    if version == 1 and specs != LEGACY_TASKS:
        raise ValueError("Configured tasks require version 2 target statistics")
    if version == 2 and report.get("task_specs") != task_config(specs):
        raise ValueError("Target statistics disagree with configured task supports")
    if set(report.get("tasks", {})) != set(specs):
        raise ValueError(f"Target statistics must contain exactly {tuple(specs)}")
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
