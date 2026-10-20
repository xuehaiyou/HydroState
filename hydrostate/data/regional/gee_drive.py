"""GEE selection and resumable native-image exports through Google Drive."""

import json
import logging
import shutil
import time
from datetime import date
from pathlib import Path

import pandas as pd

from hydrostate.data.gee.catalog import catalog
from hydrostate.data.gee.contracts import atomic_json, fingerprint, locked
from hydrostate.data.gee.products import BANDS
from hydrostate.data.gee.selection import epoch
from hydrostate.data.input_products import LEGACY_PROFILE
from hydrostate.data.regional.common import _prepare_windows, _publish_table
from hydrostate.data.regional.temporal import regional_slots
from hydrostate.inference.planning import windows

LOG = logging.getLogger(__name__)


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
    if plan["config"].get("input_profile", LEGACY_PROFILE) != LEGACY_PROFILE:
        raise ValueError("Regional Drive currently supports the legacy L1 profile only")
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
    layout = settings.get("export_layout", "aligned_float32")
    scope = settings.get("export_scope", "selected_chunks")
    transform_job = None
    if layout == "native_v1":
        from hydrostate.data.gee.native import native_job, source_groups

        metadata = {}

        def transform_job(job):
            asset = job["scene"]["asset_id"]
            if asset not in metadata:
                metadata[asset] = source_groups(ee, root, job["scene"])
            return native_job(job, metadata[asset])

    LOG.info("Regional export layout: %s; scope: %s", layout, scope)
    max_pending_dates = settings.get("max_pending_dates", 4)
    LOG.info(
        "Regional packing: max_export_gib=%.2f, max_padding_fraction=%.2f, max_pending_dates=%d",
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
                if (
                    job["contract"] != cache.contract
                    or job["grid"] != cache.grid
                    or job["chunk"] != cache.chunk
                ):
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
                        intervals = regional_slots(when, plan["config"])
                        first, last = intervals[0][0], intervals[-1][1]
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
                            when,
                            candidate_count,
                            len(todo),
                            len(pending_dates),
                        )
                        table = _prepare_windows(
                            root,
                            plan,
                            period,
                            scenes,
                            quality,
                            workers,
                            min_free_gib,
                            publish=False,
                        )
                    demands = selected_chunks(cache, table.to_dict("records"))
                    reserved = {}
                    pending_exports = state.pending()
                    for record in pending_exports:
                        job = record["payload"]
                        reserved.setdefault(job["scene"]["asset_id"], set()).update(
                            map(tuple, job["chunks"])
                        )
                    completed = {d["scene"]["asset_id"]: cache.done(d["scene"]) for d in demands}
                    missing = sum(
                        int(
                            not completed[d["scene"]["asset_id"]][c]
                            and c not in reserved.get(d["scene"]["asset_id"], set())
                        )
                        for d in demands
                        for c in d["chunks"]
                    )
                    byte_limit = settings["max_export_gib"] * 1024**3
                    if scope == "scene_region" and missing:
                        # Bound each rectangle by actual available space. The queue
                        # separately accounts for all in-flight reservations.
                        reclaimable = sum(
                            int(f["size"]) for record in pending_exports for f in record["files"]
                        )
                        byte_limit = min(
                            byte_limit,
                            (
                                transport.available_bytes()
                                + reclaimable
                                - settings["drive_reserve_gib"] * 1024**3
                            )
                            / 1.15,
                            (shutil.disk_usage(root).free - min_free_gib * 1024**3) / 2,
                        )
                        if byte_limit <= 0:
                            raise OSError("No Drive/scratch space for a regional export")
                        LOG.info(
                            "%s scene-region export cap %.2f GiB (configured %.2f GiB); "
                            "larger scenes will be split",
                            when,
                            byte_limit / 1024**3,
                            settings["max_export_gib"],
                        )
                    jobs = list(
                        selected_export_jobs(
                            cache,
                            demands,
                            max_export_bytes=byte_limit,
                            max_padding_fraction=padding,
                            reserved_chunks=reserved,
                            job_transform=transform_job,
                            export_scope=scope,
                        )
                    )
                    report = dict(
                        time=when,
                        status="selected",
                        selection_reused=reused,
                        export_layout=layout,
                        export_scope=scope,
                        effective_max_export_gib=byte_limit / 1024**3,
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
                        missing_chunks=missing,
                        new_cache_chunks=sum(len(job["chunks"]) for job in jobs),
                        reserved_chunks=sum(
                            len(d["chunks"] & reserved.get(d["scene"]["asset_id"], set()))
                            for d in demands
                        ),
                        new_exports=len(jobs),
                        new_cloud_tasks=sum(len(j.get("parts", [j])) for j in jobs),
                        export_raw_bytes=sum(j["raw_bytes"] for j in jobs),
                        mask_requests=quality.requests - mask_requests,
                        mask_response_bytes=quality.response_bytes - mask_bytes,
                        selection_seconds=time.monotonic() - started,
                    )
                    atomic_json(report_path, report)
                    LOG.info(
                        "%s selected %d/%d windows, %d scenes; %d missing chunks, "
                        "%d reserved chunks, %d new exports, %.2f GiB raw",
                        when,
                        report["accepted_windows"],
                        len(table),
                        len(demands),
                        report["missing_chunks"],
                        report["reserved_chunks"],
                        len(jobs),
                        report["export_raw_bytes"] / 1024**3,
                    )
                    pending_dates[when] = dict(
                        demands=demands,
                        selection=selection,
                        report=report,
                        report_path=report_path,
                        started=time.monotonic(),
                    )
                    publish_ready()  # A date may already be fully cached.
                    yield from jobs

            # Recovery and new dates share one queue: no startup or date-level drain.
            run_jobs(transport, state, cache, jobs_by_date(), settings, on_progress=publish_ready)
            publish_ready()
            if pending_dates:
                raise RuntimeError(
                    f"Exports drained but dates remain incomplete: {list(pending_dates)}"
                )
        finally:
            state.close()
