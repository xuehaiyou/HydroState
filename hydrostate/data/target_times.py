"""UTC observation times and product intervals used as model queries.

Intervals are [start, end); instantaneous observations have start == end.
These are observation supports, not claims that every composite is a time mean.
"""

from datetime import UTC, datetime, timedelta

import numpy as np

COVER_TASKS = {"water", "trees", "grass", "flooded_vegetation", "crops",
               "shrub_and_scrub", "built", "bare", "snow_and_ice"}


def epoch_seconds(value):
    day = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if day.tzinfo is None:
        day = day.replace(tzinfo=UTC)
    return int(day.timestamp())


def scene_time_bounds(scene, *, days=None):
    if not scene:
        return np.zeros(2, dtype="int64")
    start = scene.get("interval_start")
    end = scene.get("interval_end_exclusive", scene.get("interval_end"))
    if start is not None and end is not None:
        bounds = [epoch_seconds(start), epoch_seconds(end)]
    else:
        stamp = scene.get("time_ms", scene.get("system:time_start"))
        if stamp is None:
            raise ValueError("Observation metadata lacks its actual time")
        stamp = int(stamp) // 1000
        bounds = [stamp, stamp]
        if days:
            dt = datetime.fromtimestamp(stamp, UTC)
            bounds[1] = int(min(dt + timedelta(days=days),
                                datetime(dt.year + 1, 1, 1, tzinfo=UTC)).timestamp())
    if bounds[0] <= 0 or bounds[1] < bounds[0]:
        raise ValueError("Invalid observation time interval")
    return np.asarray(bounds, dtype="int64")


def product_time_bounds(scenes, task):
    product = ("dw" if task in COVER_TASKS else "albedo" if task.startswith("albedo_")
               else "sm" if task == "soil_moisture" else "pml" if task == "et" else task)
    return scene_time_bounds(scenes.get(product), days=8 if task in {"et", "gpp"}
                             else 4 if task == "lai" else None)
