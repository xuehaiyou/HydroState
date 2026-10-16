"""Explicit task contracts shared by data, supervision, statistics and metrics."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TaskSpec:
    # pixel: a map on the input/output grid; window_mean: one scalar per window.
    # native: complete source pixels projected with area weights; pml_native is legacy ET.
    support: str = "pixel"
    grid: str = "output"
    station_index: int | None = None


DEFAULT_TASKS = {
    "water": TaskSpec("pixel", "input", 0),
    "soil_moisture": TaskSpec("window_mean", "scalar", 1),
    "et": TaskSpec("pml_native", "native", 2),
}


def resolve_tasks(tasks=None) -> dict[str, TaskSpec]:
    """Default to the original tasks; new tasks must declare their support."""
    if tasks is None:
        return dict(DEFAULT_TASKS)
    if not isinstance(tasks, dict) or not tasks:
        raise ValueError("tasks must be a nonempty mapping of names to task specifications")
    result = {}
    for name, config in tasks.items():
        if (not re.fullmatch(r"[a-z][a-z0-9_]*", name)
                or name.endswith(("_valid", "_time")) or "_to_" in name
                or name.startswith(("site_", "et_native", "prediction_"))
                or name == "metainfo"):
            raise ValueError(f"Invalid task name: {name!r}")
        if isinstance(config, TaskSpec):
            spec = config
        else:
            config = dict(config)
            if name not in DEFAULT_TASKS and "support" not in config:
                raise ValueError(f"New task {name!r} must declare its observation support")
            defaults = asdict(DEFAULT_TASKS[name]) if name in DEFAULT_TASKS else {}
            spec = TaskSpec(**(defaults | config))
        if spec.support not in {"pixel", "window_mean", "native", "pml_native"}:
            raise ValueError(f"Unknown support for {name}: {spec.support}")
        if name in DEFAULT_TASKS and spec.support != DEFAULT_TASKS[name].support:
            raise ValueError(f"Changing built-in support for {name} requires a new task name")
        if spec.support == "pixel" and spec.grid not in {"input", "output"}:
            raise ValueError(f"Pixel task {name} requires grid=input or output")
        if spec.support == "window_mean" and spec.grid != "scalar":
            raise ValueError(f"Window-mean task {name} requires grid=scalar")
        if spec.support == "pml_native" and (name != "et" or spec.grid != "native"):
            raise ValueError("pml_native is the ET adapter; new native products need an adapter")
        if spec.support == "native" and spec.grid != "native":
            raise ValueError(f"Native task {name} requires grid=native")
        if spec.station_index is not None and spec.station_index not in range(3):
            raise ValueError("Legacy station_index must be 0, 1, 2 or None")
        result[name] = spec
    return result


def task_config(tasks=None) -> dict:
    return {name: asdict(spec) for name, spec in resolve_tasks(tasks).items()}
