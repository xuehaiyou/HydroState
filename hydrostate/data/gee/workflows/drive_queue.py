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
    optional = {"max_padding_fraction", "max_pending_dates", "export_layout", "export_scope"}
    if not required <= set(settings) or set(settings) - required - optional:
        raise ValueError(
            f"Download config requires {sorted(required)}; optional keys: {sorted(optional)}"
        )
    import math

    if any(
        not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0
        for k, v in settings.items()
        if k not in optional
    ):
        raise ValueError("All download settings must be finite and positive")
    padding = settings.get("max_padding_fraction", 0.25)
    if (
        isinstance(padding, bool)
        or not isinstance(padding, (int, float))
        or not math.isfinite(padding)
        or padding < 0
    ):
        raise ValueError("max_padding_fraction must be finite and nonnegative")
    for key in ("max_exports", "download_workers", "file_dimensions"):
        if not isinstance(settings[key], int):
            raise ValueError(f"{key} must be an integer")
    pending_dates = settings.get("max_pending_dates", 4)
    if isinstance(pending_dates, bool) or not isinstance(pending_dates, int) or pending_dates < 1:
        raise ValueError("max_pending_dates must be a positive integer")
    if settings.get("export_layout", "aligned_float32") not in ("aligned_float32", "native_v1"):
        raise ValueError("export_layout must be aligned_float32 or native_v1")
    if settings.get("export_scope", "selected_chunks") not in ("selected_chunks", "scene_region"):
        raise ValueError("export_scope must be selected_chunks or scene_region")
    if (settings.get("export_scope") == "scene_region"
            and settings.get("export_layout") != "native_v1"):
        raise ValueError("scene_region exports require native_v1")
    if settings["file_dimensions"] % 256 or settings["file_dimensions"] % cache.chunk:
        raise ValueError("file_dimensions must align to both 256 and the cache chunk")
    if settings["max_pending_gib"] < settings["max_export_gib"]:
        raise ValueError("Pending byte budget must accommodate one export")


def run_jobs(transport, state, cache, jobs, settings, *, on_progress=None):
    """Resume receipts before new work; keep at most max_exports live exports.

    A crash during submission uses the same EE request ID on recovery. A failed
    remote task is retained for diagnosis instead of duplicating its artifacts.
    A lazy job source may yield None to pause planning until active work advances.
    This permits bounded date lookahead without blocking task polling/recovery.
    """
    validate_settings(settings, cache)
    transport.ensure_folder()
    existing = state.pending()
    source = iter(chain((r["payload"] for r in existing), jobs))
    end = object()
    active, seen, waiting, exhausted = {}, set(), None, False

    def request(job, record):
        return job["parts"][record["part_index"]] if "parts" in job else job

    def submit(job, record):
        part = request(job, record)
        tid = transport.ee.data.newTaskId()[0]
        state.update(job["id"], task_id=tid, status="submitted", attempts=record["attempts"] + 1)
        actual = transport.submit(part, tid)
        state.update(job["id"], task_id=actual)
        LOG.info("Submitted export %s part=%s task=%s raw=%.1f MiB",
                 job["id"], part["id"], actual, part["raw_bytes"] / 1024**2)

    while active or not exhausted or waiting is not None:
        changed = False
        # Reap and commit before occupying more Drive or scratch space.
        for jid in list(active):
            record = state.get(jid)
            job = record["payload"]
            part = request(job, record)
            # Native groups are sequential within each logical job: max_exports
            # still bounds actual cloud tasks, including after a restart.
            if record["status"] == "planned":
                submit(job, record)
                changed = True
                continue
            if record["status"] == "submitted":
                status = retry(lambda: transport.status(record["task_id"]))
                phase = status["state"]
                if phase in ("UNKNOWN", "UNSUBMITTED"):
                    found = retry(lambda: transport.reconcile(part["id"]))
                    tid = found or transport.submit(part, record["task_id"])
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
                started = time.monotonic()
                try:
                    files = transport.inventory(part["id"])
                except IncompleteExport:
                    LOG.info("Export %s files are not fully published yet", jid)
                    continue
                if not files:
                    LOG.info("Export %s completed; awaiting Drive files", jid)
                    continue
                # Once observed, a blob's identity and checksum may not change.
                native = "parts" in job
                if native:
                    files = [dict(f, export_id=part["id"]) for f in files]
                earlier = [
                    f for f in record["files"]
                    if native and f["export_id"] != part["id"]
                ]
                previous = {
                    f["id"]: f for f in record["files"]
                    if not native or f["export_id"] == part["id"]
                }
                current = {f["id"]: f for f in files}
                if any(current.get(fid) != value for fid, value in previous.items()):
                    raise RuntimeError("An export artifact changed after it was recorded")
                all_files = [*earlier, *files]
                state.update(jid, files=all_files)
                required = sum(int(f["size"]) for f in all_files)
                if shutil.disk_usage(cache.root).free < cache.min_free_gib * 1024**3 + required * 2:
                    raise OSError("Insufficient scratch space for verified export ingestion")
                LOG.info("Downloading %s: %d files, %.1f MiB", jid, len(files), required / 1024**2)
                download_started = time.monotonic()
                paths = transport.fetch(part["id"], files)
                download_seconds = time.monotonic() - download_started
                ingest_started = time.monotonic()
                try:
                    if native:
                        from contextlib import ExitStack

                        import rasterio

                        from hydrostate.data.gee.native import validate_part

                        with ExitStack() as stack:
                            validate_part(part, [
                                stack.enter_context(rasterio.open(p)) for p in paths
                            ])
                        if record["part_index"] + 1 < len(job["parts"]):
                            state.update(
                                jid, status="planned", task_id=None,
                                part_index=record["part_index"] + 1,
                            )
                            LOG.info("Verified native part %s; retaining files until full commit",
                                     part["id"])
                            changed = True
                            continue
                        # Verify/recover previous local files before assembling;
                        # their remote copies remain until the entire cache commit.
                        paths = {
                            p["id"]: transport.fetch(p["id"], [
                                f for f in all_files if f["export_id"] == p["id"]
                            ]) for p in job["parts"]
                        }
                    ingest_export(cache, job, paths)
                except IncompleteExport:
                    LOG.info("Export %s awaiting remaining GeoTIFF parts", jid)
                    continue
                state.update(jid, status="committed")
                record = state.get(jid)
                LOG.info(
                    "Committed %s: %d cache chunks; download=%.2fs ingest=%.2fs "
                    "receive_total=%.2fs",
                    jid, len(job["chunks"]), download_seconds,
                    time.monotonic() - ingest_started, time.monotonic() - started,
                )
            if record["status"] == "committed":
                complete = cache.done(job["scene"])
                if any(not complete[cy, cx] for cy, cx in job["chunks"]):
                    raise RuntimeError("Committed cache chunks are missing; retaining Drive files")
                transport.cleanup(record)
                state.update(jid, status="cleaned")
                LOG.info("Cleaned verified export %s from Drive and staging", jid)
                del active[jid]
                changed = True
        if on_progress is not None:
            on_progress()
        while len(active) < settings["max_exports"] and not exhausted:
            if waiting is None:
                waiting = next(source, end)
                if waiting is end:
                    waiting = None
                    exhausted = True
                    break
                if waiting is None:
                    if not active:
                        raise RuntimeError("Job planning paused without any active exports")
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
                submit(job, record)
            active[jid] = job
            seen.add(jid)
            waiting = None
            changed = True
        if active and not changed:
            time.sleep(settings["poll_seconds"])
