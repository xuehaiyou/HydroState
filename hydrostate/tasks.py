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


LEGACY_TASKS = {
    "water": TaskSpec("pixel", "input", 0),
    "soil_moisture": TaskSpec("window_mean", "scalar", 1),
    "et": TaskSpec("pml_native", "native", 2),
}


def resolve_tasks(tasks=None) -> dict[str, TaskSpec]:
    """Resolve explicit supports; the default is the current retrieval contract."""
    from hydrostate.data.retrieval.contract import TASKS

    if tasks is None:
        tasks = TASKS
    if not isinstance(tasks, dict) or not tasks:
        raise ValueError("tasks must be a nonempty mapping of names to task specifications")
    result = {}
    order = list(TASKS) + [name for name in LEGACY_TASKS if name not in TASKS]
    order = [name for name in order if name in tasks] + sorted(set(tasks) - set(order))
    for name in order:
        config = tasks[name]
        if (not re.fullmatch(r"[a-z][a-z0-9_]*", name)
                or name.endswith(("_valid", "_time")) or "_to_" in name
                or name.startswith(("site_", "et_native", "prediction_"))
                or name == "metainfo"):
            raise ValueError(f"Invalid task name: {name!r}")
        if isinstance(config, TaskSpec):
            spec = config
        else:
            config = dict(config)
            if name not in LEGACY_TASKS and "support" not in config:
                raise ValueError(f"New task {name!r} must declare its observation support")
            defaults = asdict(LEGACY_TASKS[name]) if name in LEGACY_TASKS else {}
            spec = TaskSpec(**(defaults | config))
        if spec.support not in {"pixel", "window_mean", "tile_mean_weak", "native", "pml_native"}:
            raise ValueError(f"Unknown support for {name}: {spec.support}")
        if spec.support == "pixel" and spec.grid not in {"input", "output"}:
            raise ValueError(f"Pixel task {name} requires grid=input or output")
        if spec.support in {"window_mean", "tile_mean_weak"} and spec.grid != "scalar":
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
