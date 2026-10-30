"""Bounded computePixels acquisition using the current native product contract."""

import json
import logging
import queue
import shutil
import sqlite3
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from hydrostate.data.gee.backends.pixels import compute_pixels
from hydrostate.data.gee.client import initialize
from hydrostate.data.gee.contracts import atomic_json, locked
from hydrostate.data.regional.exports import composite_image, jobs
from hydrostate.data.regional.pipeline import publish_ready, status
from hydrostate.data.regional.plan import load_plan
from hydrostate.data.retrieval.composites import QUALITY
from hydrostate.data.retrieval.contract import INPUT_GROUPS
from hydrostate.storage.regional import commit, commit_arrays

LOG = logging.getLogger(__name__)


def validate_settings(settings):
    required = {"workers", "max_request_mib", "min_free_gib", "max_attempts"}
    if set(settings) != required or any(
        isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 < v < float("inf")
        for v in settings.values()
    ):
        raise ValueError(f"Pixel settings require positive finite values for {sorted(required)}")
    if any(type(settings[k]) is not int for k in ("workers", "max_attempts")):
        raise ValueError("Pixel worker/attempt counts must be integers")
    # Stay below the API's 48 MB uncompressed limit even when MB means decimal bytes.
    if settings["max_request_mib"] > 45:
        raise ValueError("max_request_mib must be at most 45; the GEE limit is 48 MB")


def transient_error(error):
    """Retry transport/service failures, never invalid expressions or local validation."""
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        code = getattr(getattr(error, "resp", None), "status", None)
        code = code or getattr(error, "status_code", None)
        if code is not None:
            return code in (408, 429, 500, 502, 503, 504)
        if isinstance(error, (TimeoutError, ConnectionError, OSError)):
            return True
        if type(error).__module__.startswith("ee.") and any(
            s in str(error).lower()
            for s in (
                "timed out", "timeout", "deadline exceeded", "internal error", "too many requests"
            )
        ):
            return True
        error = error.__cause__ or error.__context__
    return False


def check_drive_drained(root):
    path = Path(root) / "exports.sqlite"
    if not path.exists():
        return
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
        count = db.execute("SELECT COUNT(*) FROM exports WHERE status!='cleaned'").fetchone()[0]
    if count:
        raise RuntimeError(
            f"{count} unfinished Drive exports; run prepare --backend drive --drain first"
        )


def receive(ee, root, plan, job, settings, reuse=None):
    grid, spec = job["grid"], INPUT_GROUPS[job["group"]]
    h, w = grid["height"], grid["width"]
    if max(h, w) > 32768 or job["raw_bytes"] > settings["max_request_mib"] * 1024**2:
        raise ValueError("Pixel request exceeds grid or byte budget")
    if reuse is not None:
        return receive_reused(ee, root, plan, job, settings, reuse)
    started = time.monotonic()
    empty = not any(job["scenes"])
    if empty:
        # The catalog query succeeded and found no source images. This is not a failed RPC.
        values = np.zeros((len(job["bands"]), h, w), "f4")
    else:
        raw = compute_pixels(
            ee, composite_image(ee, job), job["bands"], grid["crs"], grid["transform"], w, h,
            attempts=settings["max_attempts"], retry_if=transient_error,
        )
        values = np.stack([raw[b] for b in job["bands"]])
    fetched = time.monotonic()
    n = len(spec.bands) + len(QUALITY)
    valid_pixels = 0
    for i, item in enumerate(job["products"]):
        value = values[i * n : (i + 1) * n]
        commit(root, plan, item, value)
        valid_pixels += int(np.count_nonzero(value[len(spec.bands)] > 0))
    return dict(
        request_id=job["id"], group=job["group"],
        products=[p["id"] for p in job["products"]],
        empty_source=empty, response_bytes=0 if empty else job["raw_bytes"],
        network_requests=int(not empty), reused_pixels=0, reused_products=0,
        valid_pixels=valid_pixels, fetch_seconds=fetched - started,
        ingest_seconds=time.monotonic() - fetched,
    )


def receive_reused(ee, root, plan, job, settings, reuse):
    """Reuse complete native blocks; batch-download blocks that lack complete coverage."""
    started = time.monotonic()
    grid, spec = job["grid"], INPUT_GROUPS[job["group"]]
    missing, reused_pixels, valid_pixels, reused_products = [], 0, 0, 0
    for i, item in enumerate(job["products"]):
        existing = reuse.read(item)
        if existing is None:
            missing.append(i)
            continue
        arrays, sources = existing
        pixels = grid["height"] * grid["width"]
        commit_arrays(root, plan, item, arrays, reuse=dict(pixels=pixels, sources=sources))
        reused_products += 1
        reused_pixels += pixels
        valid_pixels += int(arrays["valid"].sum())
    downloaded = dict(network_requests=0, response_bytes=0, valid_pixels=0)
    if missing:
        names = [*spec.bands, *QUALITY]
        selected = dict(job, products=[job["products"][i] for i in missing],
                        scenes=[job["scenes"][i] for i in missing],
                        bands=[f"t{i}_{b}" for i in range(len(missing)) for b in names],
                        raw_bytes=len(missing)*len(names)*grid["height"]*grid["width"]*4)
        downloaded = receive(ee, root, plan, selected, settings)
    return dict(
        request_id=job["id"], group=job["group"], products=[p["id"] for p in job["products"]],
        empty_source=not any(job["scenes"]), network_requests=downloaded["network_requests"],
        response_bytes=downloaded["response_bytes"], reused_pixels=reused_pixels,
        reused_products=reused_products, valid_pixels=valid_pixels+downloaded["valid_pixels"],
        fetch_seconds=time.monotonic()-started, ingest_seconds=0,
    )


def prepare(root, *, reuse_roots=None, dates=None):
    root = Path(root)
    plan = load_plan(root)
    if plan.get("input_source", {}).get("mode") == "direct_arrays":
        raise ValueError("Direct input views are read-only; use retile to add ready dates")
    selected = {p["start"] for p in plan["periods"]}
    if dates is not None:
        if not dates or not set(dates) <= selected:
            raise ValueError("Requested preparation dates must belong to the frozen plan")
        selected = set(dates)
    settings = json.loads((root / "pixels.json").read_text())
    validate_settings(settings)
    with locked(root / "prepare.lock", blocking=False):
        check_drive_drained(root)
        reuse = None
        reuse_path = root / "reuse.json"
        if reuse_roots is not None or reuse_path.exists():
            from hydrostate.data.regional.reuse import RegionalReuse

            previous = json.loads(reuse_path.read_text()) if reuse_path.exists() else None
            roots = reuse_roots if reuse_roots is not None else previous["roots"]
            if any(Path(r).resolve() == root.resolve() for r in roots):
                raise ValueError("Reuse sources must be different regional roots")
            reuse = RegionalReuse(plan, roots)
            identities = [p["identity"] for _, p in reuse.sources]
            if reuse_roots is None and previous["identities"] != identities:
                raise ValueError("A recorded reuse source plan changed")
            atomic_json(reuse_path, dict(roots=[r for r, _ in reuse.sources],
                                        identities=[p["identity"] for _, p in reuse.sources]))
        ee = initialize(plan["config"]["project"])
        workers = settings["workers"]
        inbox, stop = queue.Queue(maxsize=workers), threading.Event()
        pool = ThreadPoolExecutor(max_workers=workers + 1)
        active, known_ready = {}, set()
        started, last_report = time.monotonic(), 0.0
        progress = dict(
            backend="pixels", started_utc=datetime.now(UTC).isoformat(), plan=plan["identity"],
            settings=settings, requests=0, network_requests=0, products=0, empty_source_products=0,
            response_bytes=0, valid_pixels=0,
            reused_pixels=0, reused_products=0,
            selected_dates=sorted(selected),
        )
        receipts = root / "logs" / "pixels_receipts.jsonl"
        receipts.parent.mkdir(exist_ok=True)

        def produce():
            try:
                for job in jobs(
                    ee, root, plan, max_bytes=settings["max_request_mib"] * 1024**2,
                    dates=selected,
                ):
                    while not stop.is_set():
                        try:
                            inbox.put(job, timeout=0.2)
                            break
                        except queue.Full:
                            pass
                    if stop.is_set():
                        break
            except BaseException as error:
                return error
            return None

        def report(phase, **extra):
            atomic_json(
                root / "pixels_progress.json",
                dict(progress, status=phase, elapsed_seconds=time.monotonic() - started,
                     updated_utc=datetime.now(UTC).isoformat(), **extra),
            )

        producer = pool.submit(produce)
        exhausted = False
        try:
            report("running")
            while active or not exhausted:
                while len(active) < workers and not exhausted:
                    try:
                        job = inbox.get(timeout=0 if active else 0.2)
                    except queue.Empty:
                        if producer.done():
                            error = producer.result()
                            if error is not None:
                                raise error
                            exhausted = inbox.empty()
                        break
                    reserved = sum(j["raw_bytes"] for j in active.values()) + job["raw_bytes"]
                    free = shutil.disk_usage(root).free
                    if free < settings["min_free_gib"] * 1024**3 + 2 * reserved:
                        atomic_json(root / "capacity.json", dict(
                            blocked_by=["hpc_scratch"], scratch_free_gib=free / 1024**3,
                            min_free_gib=settings["min_free_gib"], reserved_bytes=reserved,
                        ))
                        raise OSError("hpc_scratch: insufficient space for pixel acquisition")
                    active[pool.submit(receive, ee, root, plan, job, settings, reuse)] = job
                if active:
                    completed, _ = wait(active, timeout=1, return_when=FIRST_COMPLETED)
                    for future in completed:
                        job = active.pop(future)
                        try:
                            result = future.result()
                        except Exception as error:
                            atomic_json(root / "pixels_failure.json", dict(
                                job=job, error=str(error), time=datetime.now(UTC).isoformat(),
                            ))
                            raise
                        with receipts.open("a") as stream:
                            stream.write(json.dumps(dict(
                                result, completed_utc=datetime.now(UTC).isoformat()
                            )) + "\n")
                        progress["requests"] += 1
                        progress["network_requests"] += result["network_requests"]
                        progress["products"] += len(result["products"])
                        progress["empty_source_products"] += (
                            len(result["products"]) if result["empty_source"] else 0
                        )
                        for key in (
                            "response_bytes", "valid_pixels", "reused_pixels", "reused_products"
                        ):
                            progress[key] += result[key]
                        known_ready.update(result["products"])
                        LOG.info("Pixels committed %s", result)
                if time.monotonic() - last_report >= 10:
                    publish_ready(root, plan, known_ready)
                    report("running")
                    last_report = time.monotonic()
            publish_ready(root, plan, known_ready)
            result = status(root)
            if any(not (root / "manifests" / (day + ".json")).exists() for day in selected):
                raise RuntimeError(f"Incomplete regional inputs: {result}")
            if result["ready_dates"] == result["planned_dates"]:
                atomic_json(root / "ready.json", result)
            report("complete", ready_dates=result["ready_dates"])
            return result
        except BaseException as error:
            report("failed", error=str(error))
            raise
        finally:
            stop.set()
            pool.shutdown(wait=True, cancel_futures=True)
