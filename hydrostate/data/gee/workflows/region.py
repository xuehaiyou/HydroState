"""Input-only Earth Engine retrieval with a reusable, spatially chunked scene cache."""

from __future__ import annotations

import json
import logging
import os
import shutil
import sqlite3
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from hydrostate.data.gee.backends.pixels import PixelSceneCache as SceneCache
from hydrostate.data.gee.catalog import catalog
from hydrostate.data.gee.client import retry as retry
from hydrostate.data.gee.contracts import atomic_json, fingerprint, locked
from hydrostate.data.gee.products import (
    BANDS,
    slots,
)
from hydrostate.data.gee.products import normalize_pixels as normalize_pixels
from hydrostate.data.gee.selection import epoch, select_window
from hydrostate.data.input_products import L2_PROFILE, LEGACY_PROFILE
from hydrostate.inference.planning import (
    windows,
)
from hydrostate.storage.schema import Schema

LOG = logging.getLogger(__name__)


def prepare_date(root, plan, period, *, workers=4, min_free_gib=5):
    if plan['config'].get('input_profile', LEGACY_PROFILE) != LEGACY_PROFILE:
        raise ValueError('Regional pixels currently support the legacy L1 profile only')
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
        if (root / "recency_update_pending.json").exists():
            raise RuntimeError(
                "Resume the interrupted regional recency revision before preparation"
            )
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
    l2 = cache.plan['config'].get('input_profile') == L2_PROFILE
    if l2:
        del values['landsat']
        del values['landsat_valid']
        values['landsat_l2'] = np.zeros((4, 7, 120, 120), 'float32')
        values['landsat_l2_valid'] = np.zeros((4, 7, 120, 120), bool)
        values['landsat_l2_time'] = np.zeros(4, 'int64')
    selected = json.loads(row["selections"])
    for slot in range(4):
        for mi, key in enumerate(BANDS):
            scene = selected[f"{key}_{slot}"]
            if scene is None or (drop_s2 and key == "s2"):
                continue
            image, valid = cache.read(scene, row)
            name = 'landsat_l2' if l2 and key == 'landsat' else key
            values[name][slot] = image
            values[f"{name}_valid"][slot] = valid
            if name == 'landsat_l2':
                values['landsat_l2_time'][slot] = scene['time']
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
    """Pipeline dates through one bounded queue and publish only complete inputs."""
    if plan['config'].get('input_profile', LEGACY_PROFILE) != LEGACY_PROFILE:
        raise ValueError('Regional Drive currently supports the legacy L1 profile only')
    from hydrostate.data.gee.backends.drive import DriveTransport
    from hydrostate.data.gee.backends.pixels import PixelQualityCache
    from hydrostate.data.gee.client import initialize
    from hydrostate.data.gee.region_plan import selected_chunks, selected_export_jobs
    from hydrostate.data.gee.state import ExportState
    from hydrostate.data.gee.workflows.drive_queue import run_jobs, validate_settings
    from hydrostate.storage.scene_cache import SceneCache as OfflineSceneCache

    root = Path(root)
    ee = initialize(plan["config"]["gee_project"])
    cache = OfflineSceneCache(root, plan, min_free_gib=min_free_gib)
    validate_settings(settings, cache)
    padding = settings.get("max_padding_fraction", 0.25)
    max_pending_dates = settings.get("max_pending_dates", 4)
    LOG.info(
        "Regional packing: max_export_gib=%.2f, max_padding_fraction=%.2f, "
        "max_pending_dates=%d",
        settings["max_export_gib"],
        padding,
        max_pending_dates,
    )
    transport = DriveTransport(
        ee,
        root,
        settings,
        token=drive_token,
        quota_project=quota_project or plan["config"]["gee_project"],
    )
    with locked(root / "prepare.lock", blocking=False):
        if (root / "recency_update_pending.json").exists():
            raise RuntimeError(
                "Resume the interrupted regional recency revision before preparation"
            )
        contract_path = root / "selection_contract.json"
        contract = dict(plan=fingerprint(plan), preprocessing=cache.contract, selection_version=1)
        if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
            raise ValueError("Selection plan changed; use a separate preparation root")
        atomic_json(contract_path, contract)
        state = ExportState(root / "exports.sqlite")
        try:
            for record in state.pending():
                job = record["payload"]
                if (job["contract"] != cache.contract or job["grid"] != cache.grid
                        or job["chunk"] != cache.chunk):
                    raise ValueError("Pending export has a different cache contract")
            quality = PixelQualityCache(cache, ee=ee)
            todo = windows(plan)
            pending_dates = {}
            (root / "reports").mkdir(exist_ok=True)

            def publish_ready():
                # Reuse each scene bitmap within this poll, but reread after commits.
                done = {}
                for when, entry in list(pending_dates.items()):
                    complete = True
                    for demand in entry["demands"]:
                        scene = demand["scene"]
                        if scene["asset_id"] not in done:
                            done[scene["asset_id"]] = cache.done(scene)
                        bitmap = done[scene["asset_id"]]
                        if any(not bitmap[c] for c in demand["chunks"]):
                            complete = False
                            break
                    if not complete:
                        continue
                    report = entry["report"]
                    report["mask_verified_chunks"] = quality.verify_full(entry["demands"])
                    dest = root / "manifests" / f"{when}.parquet"
                    _publish_table(pd.read_parquet(entry["selection"]), dest)
                    report.update(
                        status="ready",
                        transfer_wait_seconds=time.monotonic() - entry["started"],
                    )
                    atomic_json(entry["report_path"], report)
                    del pending_dates[when]
                    LOG.info("%s inputs ready: %s", when, dest)

            def jobs_by_date():
                for period in periods:
                    when = period["time"]
                    dest = root / "manifests" / f"{when}.parquet"
                    if dest.exists():
                        if len(pd.read_parquet(dest)) != len(todo):
                            raise ValueError("Incomplete date manifest")
                        continue
                    while len(pending_dates) >= max_pending_dates:
                        # Return control to polling; None does not exhaust the source.
                        yield None
                    quality.refresh_full()
                    started = time.monotonic()
                    mask_requests, mask_bytes = quality.requests, quality.response_bytes
                    selection = root / "selections" / f"{when}.parquet"
                    report_path = root / "reports" / f"preparation_{when}.json"
                    reused = selection.exists()
                    previous = json.loads(report_path.read_text()) if report_path.exists() else {}
                    if reused:
                        table = pd.read_parquet(selection)
                        if len(table) != len(todo):
                            raise ValueError("Incomplete date selection")
                        candidate_count = previous.get("candidate_scenes")
                        LOG.info("%s reusing saved selection: %d windows", when, len(table))
                    else:
                        first, last = slots(when)[0][0], slots(when)[-1][1]
                        a, b = epoch(first), epoch(last)
                        years = range(
                            date.fromisoformat(first).year, date.fromisoformat(last).year + 1
                        )
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
                        candidate_count = sum(len(group) for group in scenes.values())
                        LOG.info(
                            "%s region preparation: %d candidate scenes, %d windows; "
                            "%d earlier dates pending",
                            when, candidate_count, len(todo), len(pending_dates),
                        )
                        table = _prepare_windows(
                            root, plan, period, scenes, quality, workers, min_free_gib,
                            publish=False,
                        )
                    demands = selected_chunks(cache, table.to_dict("records"))
                    reserved = {}
                    for record in state.pending():
                        job = record["payload"]
                        reserved.setdefault(job["scene"]["asset_id"], set()).update(
                            map(tuple, job["chunks"])
                        )
                    jobs = list(selected_export_jobs(
                        cache, demands,
                        max_export_bytes=settings["max_export_gib"] * 1024**3,
                        max_padding_fraction=padding,
                        reserved_chunks=reserved,
                    ))
                    report = dict(
                        time=when, status="selected", selection_reused=reused,
                        packing=dict(
                            max_export_gib=settings["max_export_gib"],
                            max_padding_fraction=padding,
                        ),
                        max_pending_dates=max_pending_dates,
                        candidate_scenes=candidate_count,
                        selected_scenes=len(demands),
                        windows=len(table),
                        accepted_windows=int(table["accepted"].sum()),
                        required_chunks=sum(len(d["chunks"]) for d in demands),
                        missing_chunks=sum(len(job["chunks"]) for job in jobs),
                        reserved_chunks=sum(
                            len(d["chunks"] & reserved.get(d["scene"]["asset_id"], set()))
                            for d in demands
                        ),
                        new_exports=len(jobs),
                        export_raw_bytes=sum(j["raw_bytes"] for j in jobs),
                        mask_requests=quality.requests - mask_requests,
                        mask_response_bytes=quality.response_bytes - mask_bytes,
                        selection_seconds=time.monotonic() - started,
                    )
                    atomic_json(report_path, report)
                    LOG.info(
                        "%s selected %d/%d windows, %d scenes; %d missing chunks, "
                        "%d reserved chunks, %d new exports, %.2f GiB raw",
                        when, report["accepted_windows"], len(table), len(demands),
                        report["missing_chunks"], report["reserved_chunks"], len(jobs),
                        report["export_raw_bytes"] / 1024**3,
                    )
                    pending_dates[when] = dict(
                        demands=demands, selection=selection, report=report,
                        report_path=report_path, started=time.monotonic(),
                    )
                    publish_ready()  # A date may already be fully cached.
                    yield from jobs

            # Recovery and new dates share one queue: no startup or date-level drain.
            run_jobs(
                transport, state, cache, jobs_by_date(), settings, on_progress=publish_ready
            )
            publish_ready()
            if pending_dates:
                raise RuntimeError(
                    f"Exports drained but dates remain incomplete: {list(pending_dates)}"
                )
        finally:
            state.close()
