"""Shared window selection, durable manifests and model input assembly."""

import json
import logging
import os
import shutil
import sqlite3
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

import numpy as np
import pandas as pd

from hydrostate.data.gee.products import BANDS
from hydrostate.data.gee.selection import select_window
from hydrostate.data.input_products import L2_PROFILE
from hydrostate.inference.planning import windows
from hydrostate.legacy.schema import LegacySchema as Schema

LOG = logging.getLogger(__name__)


def _publish_table(table, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = dest.with_suffix(".parquet.part")
    table.to_parquet(temporary, index=False)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, dest)


def _prepare_windows(root, plan, period, scenes, cache, workers, min_free_gib, *, publish=True):
    dest = root / ("manifests" if publish else "selections") / f"{period['time']}.parquet"
    todo = windows(plan)
    db = sqlite3.connect(root / "preparation.sqlite")
    try:
        db.execute(
            "CREATE TABLE IF NOT EXISTS windows (date TEXT, id TEXT, result TEXT, "
            "PRIMARY KEY(date,id))"
        )
        saved = {
            wid: json.loads(value)
            for wid, value in db.execute(
                "SELECT id,result FROM windows WHERE date=?", (period["time"],)
            )
        }
        pending = [w for w in todo if w["window_id"] not in saved]
        accepted_count = sum(r["accepted"] for r in saved.values())
        LOG.info(
            "%s resuming %d/%d windows; free=%.1f GiB, reserve=%.1f GiB",
            period["time"],
            len(saved),
            len(todo),
            shutil.disk_usage(root).free / 1024**3,
            min_free_gib,
        )
        with ThreadPoolExecutor(max_workers=workers) as pool:
            remaining = iter(pending)
            futures = set()

            def submit_next():
                window = next(remaining, None)
                if window is not None:
                    futures.add(pool.submit(select_window, plan, period, window, scenes, cache))

            for _ in range(workers):
                submit_next()
            try:
                while futures:
                    completed, futures = wait(futures, return_when=FIRST_COMPLETED)
                    for future in completed:
                        result = future.result()
                        saved[result["window_id"]] = result
                        accepted_count += result["accepted"]
                        db.execute(
                            "INSERT INTO windows VALUES (?,?,?)",
                            (period["time"], result["window_id"], json.dumps(result)),
                        )
                        db.commit()
                        if len(saved) == len(todo) or len(saved) % max(1, len(todo) // 100) == 0:
                            LOG.info(
                                "%s prepared %d/%d windows; accepted=%d",
                                period["time"],
                                len(saved),
                                len(todo),
                                accepted_count,
                            )
                    for _ in completed:
                        submit_next()
            except BaseException:
                cache.stopped.set()
                for future in futures:
                    future.cancel()
                raise
        table = pd.DataFrame([saved[w["window_id"]] for w in todo])
        _publish_table(table, dest)
        return table
    finally:
        db.close()


def assemble_inputs(cache, row, *, drop_s2=False):
    values = {name: np.zeros(spec.tail_shape, spec.dtype) for name, spec in Schema().inputs.items()}
    l2 = cache.plan["config"].get("input_profile") == L2_PROFILE
    if l2:
        del values["landsat"]
        del values["landsat_valid"]
        values["landsat_l2"] = np.zeros((4, 7, 120, 120), "float32")
        values["landsat_l2_valid"] = np.zeros((4, 7, 120, 120), bool)
        values["landsat_l2_time"] = np.zeros(4, "int64")
    selected = json.loads(row["selections"])
    for slot in range(4):
        for mi, key in enumerate(BANDS):
            scene = selected[f"{key}_{slot}"]
            if scene is None or (drop_s2 and key == "s2"):
                continue
            image, valid = cache.read(scene, row)
            name = "landsat_l2" if l2 and key == "landsat" else key
            values[name][slot] = image
            values[f"{name}_valid"][slot] = valid
            if name == "landsat_l2":
                values["landsat_l2_time"][slot] = scene["time"]
            values["sensor_valid"][slot, mi] = bool(valid.any())
            values["observation_time"][slot, mi] = scene["time"]
    return values
