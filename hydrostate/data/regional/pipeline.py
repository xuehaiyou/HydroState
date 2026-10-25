"""Bounded Drive production with independent planning and download/ingestion."""

import json
import logging
import queue
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from hydrostate.data.gee.backends.drive import IncompleteExport, resume_download
from hydrostate.data.gee.client import initialize
from hydrostate.data.gee.contracts import atomic_json, locked
from hydrostate.data.gee.state import ExportState
from hydrostate.data.regional.exports import CompositeDriveTransport, jobs
from hydrostate.data.regional.plan import load_plan
from hydrostate.storage.regional import ingest, ready

LOG = logging.getLogger(__name__)


def validate_settings(settings):
    required = {
        "max_exports",
        "download_workers",
        "file_dimensions",
        "max_export_gib",
        "max_pending_gib",
        "drive_reserve_gib",
        "min_free_gib",
        "poll_seconds",
    }
    if set(settings) != required or any(
        isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 < v < float("inf")
        for v in settings.values()
    ):
        raise ValueError(f"Drive settings require positive finite values for {sorted(required)}")
    for key in ("max_exports", "download_workers", "file_dimensions"):
        if type(settings[key]) is not int:
            raise ValueError(f"{key} must be an integer")
    if (
        settings["file_dimensions"] % 256
        or settings["max_pending_gib"] < settings["max_export_gib"]
    ):
        raise ValueError("Invalid export dimensions or pending budget")


def capacity(settings, raw_bytes, reserved, drive_free, scratch_free):
    """Report each limiting resource separately; never infer that Drive is full."""
    gib = 1024**3
    need = raw_bytes + reserved
    values = dict(
        drive_free_gib=drive_free / gib,
        scratch_free_gib=scratch_free / gib,
        planned_gib=need / gib,
        drive_required_gib=(need * 1.15) / gib + settings["drive_reserve_gib"],
        scratch_required_gib=need * 2 / gib + settings["min_free_gib"],
    )
    failures = []
    if need > settings["max_pending_gib"] * gib:
        failures.append("pending_export_budget")
    if drive_free < need * 1.15 + settings["drive_reserve_gib"] * gib:
        failures.append("google_drive")
    if scratch_free < need * 2 + settings["min_free_gib"] * gib:
        failures.append("hpc_scratch")
    return dict(values, blocked_by=failures)


def receive(transport, root, plan, job, files, settings):
    from google.auth.transport.requests import AuthorizedSession

    needed = sum(int(f["size"]) for f in files)
    if shutil.disk_usage(root).free < settings["min_free_gib"] * 1024**3 + 2 * needed:
        raise OSError("hpc_scratch: insufficient room for verified ingestion")
    started = time.monotonic()
    paths = []
    # One session per worker; no thread-unsafe shared HTTP client or nested pool.
    with AuthorizedSession(transport.credentials) as session:
        for remote in files:
            path = transport.local_path(job["id"], remote)
            resume_download(session, remote, path)
            paths.append(path)
    downloaded = time.monotonic()
    ingest(root, plan, job, paths)
    return dict(
        download_seconds=downloaded - started,
        ingest_seconds=time.monotonic() - downloaded,
        bytes=needed,
    )


def publish_ready(root, plan, known_ready=None):
    root = Path(root)
    folder = root / "manifests"
    folder.mkdir(exist_ok=True)
    known_ready = set() if known_ready is None else known_ready

    def is_ready(item):
        if item["id"] in known_ready:
            return True
        if ready(root, item):
            known_ready.add(item["id"])
            return True
        return False

    for period in plan["periods"]:
        source = root / "targets" / (period["start"] + ".json")
        dest = folder / source.name
        if dest.exists() or not source.exists():
            continue
        target = json.loads(source.read_text())
        if all(is_ready(item) for item in target["products"]):
            atomic_json(dest, target)
            LOG.info("Inputs ready: %s", period["start"])


def status(root, *, audit=False):
    root = Path(root)
    plan = load_plan(root)
    completed, products = [], {}
    for p in plan["periods"]:
        path = root / "manifests" / (p["start"] + ".json")
        if path.exists():
            record = json.loads(path.read_text())
            if record["plan"] != plan["identity"]:
                raise ValueError("Manifest belongs to a different plan")
            completed.append(p["start"])
            products.update({r["id"]: r for r in record["products"]})
    if audit:
        import numpy as np
        import zarr

        from hydrostate.storage.regional import product_path

        for item in products.values():
            if not ready(root, item):
                raise ValueError(f"Missing product: {item['id']}")
            group = zarr.open_group(str(product_path(root, item)), mode="r")
            valid = group["valid"][:].astype(bool)
            if not np.isfinite(group["values"][:]).all() or not np.array_equal(
                valid[0], group["quality"][0] > 0
            ):
                raise ValueError(f"Invalid product: {item['id']}")
    return dict(
        planned_dates=len(plan["periods"]),
        ready_dates=len(completed),
        dates=completed,
        unique_products=len(products),
        audited=audit,
    )


def prepare(root, *, token=None, retry_failed=False):
    root = Path(root)
    plan = load_plan(root)
    settings = json.loads((root / "drive.json").read_text())
    validate_settings(settings)
    ee = initialize(plan["config"]["project"])
    transport = CompositeDriveTransport(
        ee, root, settings, token=token, quota_project=plan["config"]["project"]
    )
    with locked(root / "prepare.lock", blocking=False):
        transport.ensure_folder()
        state = ExportState(root / "exports.sqlite")
        inbox, stop = queue.Queue(maxsize=2), threading.Event()
        pool = ThreadPoolExecutor(max_workers=settings["download_workers"] + 1)
        futures, active, waiting = {}, {}, None
        known_ready = set()
        try:
            pending = state.pending()
            for record in pending:
                if record["status"] == "failed":
                    if not retry_failed:
                        raise RuntimeError(f"Failed export {record['id']}: {record['error']}")
                    state.update(record["id"], status="planned", task_id=None, error=None)
            reserved = {p["id"] for r in pending for p in r["payload"]["products"]}

            def produce():
                try:
                    for job in jobs(ee, root, plan, settings, reserved):
                        while not stop.is_set():
                            try:
                                inbox.put(job, timeout=0.2)
                                break
                            except queue.Full:
                                pass
                        if stop.is_set():
                            return
                except BaseException as exc:
                    return exc
                return None

            producer = pool.submit(produce)
            recovery = iter(r["payload"] for r in pending)
            exhausted_recovery = exhausted = False
            while active or waiting is not None or not exhausted:
                changed = False
                for jid in list(active):
                    record, job = state.get(jid), active[jid]
                    if jid in futures:
                        if not futures[jid].done():
                            continue
                        timing = futures.pop(jid).result()
                        state.update(jid, status="committed")
                        record = state.get(jid)
                        LOG.info("Committed %s %s", jid, timing)
                        changed = True
                    if record["status"] == "submitted":
                        remote = transport.status(record["task_id"])
                        phase = remote["state"]
                        if phase in ("UNKNOWN", "UNSUBMITTED"):
                            # Recover the same request identity after ambiguous submission.
                            transport.submit(job, record["task_id"])
                        elif phase in ("FAILED", "CANCELLED"):
                            state.update(jid, status="failed", error=str(remote))
                            raise RuntimeError(f"GEE export {jid}: {remote}")
                        elif phase == "COMPLETED":
                            state.update(jid, status="exported")
                            record = state.get(jid)
                    if (
                        record["status"] == "exported"
                        and len(futures) < settings["download_workers"]
                    ):
                        try:
                            files = transport.inventory(jid)
                        except IncompleteExport:
                            continue
                        if not files:
                            continue
                        previous = {f["id"]: f for f in record["files"]}
                        current = {f["id"]: f for f in files}
                        if any(current.get(k) != v for k, v in previous.items()):
                            raise ValueError("Published Drive artifact changed")
                        state.update(jid, files=files)
                        futures[jid] = pool.submit(
                            receive, transport, root, plan, job, files, settings
                        )
                        changed = True
                    if record["status"] == "committed":
                        if not all(ready(root, p) for p in job["products"]):
                            raise ValueError("Committed products missing; preserving Drive copies")
                        transport.cleanup(record)
                        state.update(jid, status="cleaned")
                        del active[jid]
                        changed = True
                if changed:
                    publish_ready(root, plan, known_ready)
                while len(active) < settings["max_exports"] and not exhausted:
                    if waiting is None:
                        if not exhausted_recovery:
                            waiting = next(recovery, None)
                            exhausted_recovery = waiting is None
                        if waiting is None:
                            try:
                                waiting = inbox.get_nowait()
                            except queue.Empty:
                                if producer.done():
                                    error = producer.result()
                                    if error is not None:
                                        raise error
                                    # Producer may enqueue its final job between get_nowait()
                                    # and done(); once finished, this check is stable.
                                    exhausted = inbox.empty()
                                break
                    job = waiting
                    state.add(job)
                    record = state.get(job["id"])
                    if record["status"] == "cleaned":
                        waiting = None
                        continue
                    if record["status"] == "planned":
                        budget = capacity(
                            settings,
                            job["raw_bytes"],
                            sum(j["raw_bytes"] for j in active.values()),
                            transport.available_bytes(),
                            shutil.disk_usage(root).free,
                        )
                        if budget["blocked_by"]:
                            atomic_json(root / "capacity.json", budget)
                            if not active:
                                raise OSError(f"Export capacity blocked: {budget}")
                            break
                        tid = ee.data.newTaskId()[0]
                        state.update(
                            job["id"],
                            status="submitted",
                            task_id=tid,
                            attempts=record["attempts"] + 1,
                        )
                        actual = transport.submit(job, tid)
                        state.update(job["id"], task_id=actual)
                        LOG.info("Submitted %s: %d slots", job["id"], len(job["products"]))
                    active[job["id"]] = job
                    waiting = None
                    changed = True
                if not changed:
                    time.sleep(min(settings["poll_seconds"], 1 if not active else 30))
            publish_ready(root, plan, known_ready)
            # Each block was checked at commit. Full rereads are an explicit audit command.
            report = status(root)
            if report["ready_dates"] != report["planned_dates"]:
                raise RuntimeError(f"Incomplete regional inputs: {report}")
            atomic_json(root / "ready.json", report)
            return report
        finally:
            stop.set()
            pool.shutdown(wait=True, cancel_futures=True)
            state.close()
