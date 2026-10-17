"""Input-only Earth Engine retrieval with a reusable, spatially chunked scene cache."""

from __future__ import annotations

import json
import logging
import os
import shutil
import sqlite3
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from hydrostate.data.gee.backends.pixels import PixelSceneCache as SceneCache
from hydrostate.data.gee.catalog import catalog
from hydrostate.data.gee.contracts import locked
from hydrostate.data.gee.products import (
    BANDS,
    slots,
)
from hydrostate.data.gee.products import normalize_pixels as normalize_pixels
from hydrostate.data.gee.selection import epoch, select_window
from hydrostate.inference.planning import (
    windows,
)
from hydrostate.storage.schema import Schema

LOG = logging.getLogger(__name__)


def prepare_date(root, plan, period, *, workers=4, min_free_gib=5):
    import ee

    root = Path(root)
    ee.Initialize(project=plan["config"]["gee_project"])
    ee.data.setDeadline(180000)
    dest = root / "manifests" / f"{period['time']}.parquet"
    if dest.exists():
        table = pd.read_parquet(dest)
        if len(table) != len(windows(plan)):
            raise ValueError("Incomplete date manifest")
        return table
    with locked(root / "prepare.lock", blocking=False):
        t = date.fromisoformat(period["time"])
        years = range((t - timedelta(days=63)).year, t.year + 1)
        scenes = {
            key: sorted(
                [s for year in years for s in catalog(root, plan, key, year, ee)],
                key=lambda s: (s["time"], s["asset_id"]),
                reverse=True,
            )
            for key in BANDS
        }
        cache = SceneCache(root, plan, ee=ee, min_free_gib=min_free_gib)
        return _prepare_windows(root, plan, period, scenes, cache, workers, min_free_gib)


def _prepare_windows(root, plan, period, scenes, cache, workers, min_free_gib):
    dest = root / "manifests" / f"{period['time']}.parquet"
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
        dest.parent.mkdir(parents=True, exist_ok=True)
        temporary = dest.with_suffix(".parquet.part")
        table.to_parquet(temporary, index=False)
        os.replace(temporary, dest)
        return table
    finally:
        db.close()


def assemble_inputs(cache, row, *, drop_s2=False):
    values = {name: np.zeros(spec.tail_shape, spec.dtype) for name, spec in Schema().inputs.items()}
    selected = json.loads(row["selections"])
    for slot in range(4):
        for mi, key in enumerate(BANDS):
            scene = selected[f"{key}_{slot}"]
            if scene is None or (drop_s2 and key == "s2"):
                continue
            image, valid = cache.read(scene, row)
            values[key][slot] = image
            values[f"{key}_valid"][slot, 0] = valid
            values["sensor_valid"][slot, mi] = bool(valid.any())
            values["observation_time"][slot, mi] = scene["time"]
    return values


def prepare_region_drive(
    root,
    plan,
    periods,
    settings,
    *,
    workers=4,
    min_free_gib=20,
    drive_token=None,
    quota_project=None,
):
    """Fetch regional scene rectangles, then apply the unchanged window selection."""
    from hydrostate.data.gee.backends.drive import DriveTransport
    from hydrostate.data.gee.client import initialize
    from hydrostate.data.gee.region_plan import export_jobs
    from hydrostate.data.gee.state import ExportState
    from hydrostate.data.gee.workflows.drive_queue import run_jobs, validate_settings
    from hydrostate.storage.scene_cache import SceneCache as OfflineSceneCache

    root = Path(root)
    ee = initialize(plan["config"]["gee_project"])
    cache = OfflineSceneCache(root, plan, min_free_gib=min_free_gib)
    validate_settings(settings, cache)
    transport = DriveTransport(
        ee,
        root,
        settings,
        token=drive_token,
        quota_project=quota_project or plan["config"]["gee_project"],
    )
    with locked(root / "prepare.lock", blocking=False):
        state = ExportState(root / "exports.sqlite")
        try:
            # Recover previous submissions/commits even if their date manifest exists.
            run_jobs(transport, state, cache, [], settings)
            todo = windows(plan)
            for period in periods:
                dest = root / "manifests" / f"{period['time']}.parquet"
                if dest.exists():
                    if len(pd.read_parquet(dest)) != len(todo):
                        raise ValueError("Incomplete date manifest")
                    continue
                first, last = slots(period["time"])[0][0], slots(period["time"])[-1][1]
                a, b = epoch(first), epoch(last)
                years = range(date.fromisoformat(first).year, date.fromisoformat(last).year + 1)
                scenes = {
                    key: sorted(
                        [
                            s
                            for year in years
                            for s in catalog(root, plan, key, year, ee)
                            if a <= s["time"] < b
                        ],
                        key=lambda s: (s["time"], s["asset_id"]),
                        reverse=True,
                    )
                    for key in BANDS
                }
                candidates = [s for group in scenes.values() for s in group]
                LOG.info(
                    "%s region preparation: %d candidate scenes, %d windows",
                    period["time"],
                    len(candidates),
                    len(todo),
                )
                jobs = export_jobs(
                    cache, candidates, todo, max_export_bytes=settings["max_export_gib"] * 1024**3
                )
                run_jobs(transport, state, cache, jobs, settings)
                _prepare_windows(root, plan, period, scenes, cache, workers, min_free_gib)
        finally:
            state.close()
