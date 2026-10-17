"""Bounded, restartable regional export queue; no inference or sample selection."""

import logging
import shutil
import time
from itertools import chain

from hydrostate.data.gee.backends.drive import IncompleteExport, ingest_export
from hydrostate.data.gee.client import retry

LOG = logging.getLogger(__name__)


def validate_settings(settings, cache):
    required = {
        "max_exports",
        "download_workers",
        "file_dimensions",
        "max_export_gib",
        "max_pending_gib",
        "drive_reserve_gib",
        "poll_seconds",
    }
    if set(settings) != required:
        raise ValueError(f"Download config must contain exactly {sorted(required)}")
    import math

    if any(
        not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in settings.values()
    ):
        raise ValueError("All download settings must be finite and positive")
    for key in ("max_exports", "download_workers", "file_dimensions"):
        if not isinstance(settings[key], int):
            raise ValueError(f"{key} must be an integer")
    if settings["file_dimensions"] % 256 or settings["file_dimensions"] % cache.chunk:
        raise ValueError("file_dimensions must align to both 256 and the cache chunk")
    if settings["max_pending_gib"] < settings["max_export_gib"]:
        raise ValueError("Pending byte budget must accommodate one export")


def run_jobs(transport, state, cache, jobs, settings):
    """Resume receipts before new work; keep at most max_exports live exports.

    A crash during submission uses the same EE request ID on recovery. A failed
    remote task is retained for diagnosis instead of duplicating its artifacts.
    """
    validate_settings(settings, cache)
    transport.ensure_folder()
    existing = state.pending()
    source = iter(chain((r["payload"] for r in existing), jobs))
    active, seen, waiting, exhausted = {}, set(), None, False
    while active or not exhausted or waiting is not None:
        changed = False
        # Reap and commit before occupying more Drive or scratch space.
        for jid in list(active):
            record = state.get(jid)
            job = record["payload"]
            if record["status"] == "submitted":
                status = retry(lambda: transport.status(record["task_id"]))
                phase = status["state"]
                if phase in ("UNKNOWN", "UNSUBMITTED"):
                    found = retry(lambda: transport.reconcile(jid))
                    tid = found or transport.submit(job, record["task_id"])
                    state.update(jid, task_id=tid)
                    continue
                if phase in ("READY", "RUNNING"):
                    LOG.info("Export %s %s: %s", jid, phase, job["scene"]["asset_id"])
                    continue
                if phase != "COMPLETED":
                    state.update(jid, status="failed", error=str(status))
                    raise RuntimeError(f"Regional export {jid} ended in {phase}: {status}")
                state.update(jid, status="exported")
                record = state.get(jid)
            if record["status"] == "exported":
                try:
                    files = transport.inventory(jid)
                except IncompleteExport:
                    LOG.info("Export %s files are not fully published yet", jid)
                    continue
                if not files:
                    LOG.info("Export %s completed; awaiting Drive files", jid)
                    continue
                # Once observed, a blob's identity and checksum may not change.
                previous = {f["id"]: f for f in record["files"]}
                current = {f["id"]: f for f in files}
                if any(current.get(fid) != value for fid, value in previous.items()):
                    raise RuntimeError("An export artifact changed after it was recorded")
                state.update(jid, files=files)
                required = sum(int(f["size"]) for f in files)
                if shutil.disk_usage(cache.root).free < cache.min_free_gib * 1024**3 + required * 2:
                    raise OSError("Insufficient scratch space for verified export ingestion")
                LOG.info("Downloading %s: %d files, %.1f MiB", jid, len(files), required / 1024**2)
                paths = transport.fetch(jid, files)
                try:
                    ingest_export(cache, job, paths)
                except IncompleteExport:
                    LOG.info("Export %s awaiting remaining GeoTIFF parts", jid)
                    continue
                state.update(jid, status="committed")
                record = state.get(jid)
                LOG.info("Committed %s: %d cache chunks", jid, len(job["chunks"]))
            if record["status"] == "committed":
                complete = cache.done(job["scene"])
                if any(not complete[cy, cx] for cy, cx in job["chunks"]):
                    raise RuntimeError("Committed cache chunks are missing; retaining Drive files")
                transport.cleanup(record)
                state.update(jid, status="cleaned")
                LOG.info("Cleaned verified export %s from Drive and staging", jid)
                del active[jid]
                changed = True
        while len(active) < settings["max_exports"] and not exhausted:
            if waiting is None:
                waiting = next(source, None)
                if waiting is None:
                    exhausted = True
                    break
            job = waiting
            jid = job["id"]
            if jid in seen:
                waiting = None
                continue
            state.add(job)
            record = state.get(jid)
            if record["status"] == "cleaned":
                seen.add(jid)
                waiting = None
                continue
            if record["status"] == "failed":
                raise RuntimeError(f"Unresolved export failure {jid}: {record['error']}")
            if record["status"] == "planned":
                pending = sum(j["raw_bytes"] for j in active.values())
                budget = pending + job["raw_bytes"]
                # Float GeoTIFFs may be uncompressed; reserve overhead as well.
                allowed = (
                    budget <= settings["max_pending_gib"] * 1024**3
                    and transport.available_bytes()
                    >= budget * 1.15 + settings["drive_reserve_gib"] * 1024**3
                    and shutil.disk_usage(cache.root).free
                    >= cache.min_free_gib * 1024**3 + budget * 2
                )
                if not allowed:
                    if not active:
                        raise OSError(
                            "Drive/scratch capacity below export budget; receipts retained"
                        )
                    break
                tid = transport.ee.data.newTaskId()[0]
                state.update(jid, task_id=tid, status="submitted", attempts=record["attempts"] + 1)
                actual = transport.submit(job, tid)
                state.update(jid, task_id=actual)
                LOG.info(
                    "Submitted regional export %s task=%s raw=%.1f MiB chunks=%d",
                    jid,
                    actual,
                    job["raw_bytes"] / 1024**2,
                    len(job["chunks"]),
                )
            active[jid] = job
            seen.add(jid)
            waiting = None
            changed = True
        if active and not changed:
            time.sleep(settings["poll_seconds"])
