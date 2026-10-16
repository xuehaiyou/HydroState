"""Paired live benchmark of the production Drive path and computePixels.

Uses a separate output directory on the same filesystem as production. Never
modifies source samples. Drive runs call the production scheduler unchanged;
instrumentation only records timing. Direct runs preserve the export expression,
Float32 bands, explicit grid, and production ingestion/validation.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import ee
import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from rasterio.windows import Window

from hydrostate.data import gee_sampling as gs


def save(path, value):
    gs.atomic_json(path, value)


def prepare(args):
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "selection.json").exists():
        raise FileExistsError("Selection already exists; reuse it with a run command")
    rows = pd.read_parquet(args.source / "sample_index.parquet").to_dict("records")
    rng = random.Random(20261015)
    rng.shuffle(rows)
    seen, chosen, used = set(), [], set()
    for block in ("A", "B"):
        for fb in range(7):

            def features(row):
                inputs = json.loads(row["inputs"])
                labels = json.loads(row["labels"])
                n = sum(bool(v) for v in inputs.values())
                return {
                    ("year", row["target_time"][:4]),
                    ("quarter", (int(row["target_time"][5:7]) - 1) // 3),
                    ("climate", row["climate_class"]),
                    ("latitude", int(float(row["lat"]) // 30)),
                    ("longitude", int(float(row["lon"]) // 60)),
                    ("available_inputs", n),
                    ("sm", bool(labels["sm"])),
                    ("pml", bool(labels["pml"])),
                }

            eligible = [r for r in rows if r["frequency_bin"] == fb and r["sample_id"] not in used]
            selected = max(eligible, key=lambda r: len(features(r) - seen))
            seen.update(features(selected))
            used.add(selected["sample_id"])
            chosen.append(dict(block=block, row=selected))
    save(
        args.output / "selection.json",
        {
            "source": str(args.source.resolve()),
            "project": args.project,
            "selected_at": datetime.now(UTC).isoformat(),
            "strategy": (
                "two per frequency bin; greedy diversity of date, geography, climate, availability"
            ),
            "planned_order": ["drive/A", "direct/B", "direct/A", "drive/B"],
            "samples": chosen,
        },
    )
    print(json.dumps({"samples": len(chosen), "output": str(args.output)}, indent=2))


def drive_service(args):
    if not args.drive_quota_project:
        return gs.drive_client(SimpleNamespace(drive_token=args.drive_token))
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build

    credentials = ee.data.get_persistent_credentials().with_quota_project(args.drive_quota_project)
    credentials.refresh(Request())
    return build("drive", "v3", credentials=credentials, cache_discovery=False)


def load_rows(args):
    selection = json.loads((args.output / "selection.json").read_text())
    if selection["project"] != args.project:
        raise ValueError("Project differs from saved selection")
    return [s["row"] for s in selection["samples"] if s["block"] == args.block]


def sample_from_row(row):
    sample = copy.deepcopy(row)
    for key in ("inputs", "labels", "transform"):
        sample[key] = json.loads(sample[key])
    return sample


def direct_one(row, root):
    started = time.perf_counter()
    sample = sample_from_row(row)
    sid = sample["sample_id"]
    pml = sample["labels"].get("pml")
    if pml:
        pml["projection"] = ee.Image(pml["asset_id"]).select("ET").projection().getInfo()
    expression, names = gs.export_image(ee, sample)
    prep_done = time.perf_counter()
    tr = sample["transform"]
    payload = ee.data.computePixels(
        {
            "expression": expression,
            "fileFormat": "NUMPY_NDARRAY",
            "bandIds": names,
            "grid": {
                "dimensions": {"width": gs.SIZE, "height": gs.SIZE},
                "crsCode": sample["crs"],
                "affineTransform": dict(
                    zip(("scaleX", "shearX", "translateX", "shearY", "scaleY", "translateY"), tr)
                ),
            },
        }
    )
    response_done = time.perf_counter()
    if payload.shape != (gs.SIZE, gs.SIZE) or list(payload.dtype.names or ()) != names:
        raise ValueError("computePixels returned unexpected shape or named band order")
    if any(payload.dtype[name] != np.dtype("float32") for name in names):
        raise ValueError("computePixels returned a non-Float32 band")
    data = np.stack([payload[name] for name in names])
    if not np.isfinite(data).all():
        raise ValueError("computePixels returned nonfinite values")
    path = root / "downloads" / f"{sid}.tif"
    temp = path.with_suffix(".tif.part")
    digest = hashlib.sha256(data.tobytes()).hexdigest()
    # computePixels GeoTIFF lacks band descriptions. Named NumPy fields allow
    # us to validate identity before creating a TIFF accepted by production.
    with rasterio.open(
        temp,
        "w",
        driver="GTiff",
        width=gs.SIZE,
        height=gs.SIZE,
        count=len(names),
        dtype="float32",
        crs=sample["crs"],
        transform=Affine(*tr),
        compress="lzw",
        photometric="MINISBLACK",
    ) as dst:
        dst.write(data)
        dst.descriptions = tuple(names)
    with temp.open("rb+") as handle:
        os.fsync(handle.fileno())
    with rasterio.open(temp) as persisted:
        if hashlib.sha256(persisted.read().tobytes()).hexdigest() != digest:
            raise ValueError("Local write checksum mismatch")
    os.replace(temp, path)
    disk_done = time.perf_counter()
    save(path.with_suffix(".json"), {"sample": sample, "bands": names, "pixel_sha256": digest})
    ip, lp = gs.ingest(path, sample, names, root)
    done = time.perf_counter()
    result = dict(
        sample_id=sid,
        prep_s=prep_done - started,
        compute_and_receive_s=response_done - prep_done,
        write_fsync_checksum_s=disk_done - response_done,
        ingest_s=done - disk_done,
        total_s=done - started,
        bytes=path.stat().st_size,
        decoded_payload_bytes=payload.nbytes,
        input_shard=ip,
        label_shard=lp,
        input_row=0,
        label_row=0,
    )
    save(root / f"timing_{sid}.json", result)
    logging.info(
        "Direct %s: %.2fs (compute/receive %.2fs; disk %.3fs; ingest %.2fs)",
        sid,
        result["total_s"],
        result["compute_and_receive_s"],
        result["write_fsync_checksum_s"],
        result["ingest_s"],
    )
    return result


def run_direct(args, rows, root):
    (root / "downloads").mkdir()
    started = time.perf_counter()
    # EE is initialized before starting workers. This client's transport uses
    # requests connection pooling; each worker writes a distinct shard pair.
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(pool.map(lambda row: direct_one(row, root), rows))
    all_committed_s = time.perf_counter() - started
    timings = {r["sample_id"]: r for r in results}
    manifest = [
        dict(
            row,
            **{
                k: timings[row["sample_id"]][k]
                for k in ("input_shard", "label_shard", "input_row", "label_row")
            },
        )
        for row in rows
    ]
    pd.DataFrame(manifest).to_parquet(root / "samples.parquet", index=False)
    return {
        "wall_s": time.perf_counter() - started,
        "all_committed_s": all_committed_s,
        "samples": results,
    }


def run_drive(args, rows, root):
    drive = drive_service(args)
    # Probe credentials before any exports are created.
    drive.files().list(pageSize=1, fields="files(id)").execute()
    events, metrics, statuses = [], {}, {}
    original_download, original_ingest = gs.download, gs.ingest
    original_status, original_start = ee.data.getTaskStatus, ee.batch.Task.start
    started = time.perf_counter()
    ingest_depth = 0

    def record(kind, **values):
        event = dict(kind=kind, elapsed_s=time.perf_counter() - started, **values)
        events.append(event)
        with (root / "events.jsonl").open("a") as f:
            f.write(json.dumps(event) + "\n")

    def timed_start(task):
        sid = task.config["description"].removeprefix("hs_")
        record("submit_start", sample_id=sid)
        result = original_start(task)
        record("submit_end", sample_id=sid, task_id=task.id)
        return result

    def timed_status(*a, **kw):
        result = original_status(*a, **kw)
        for state in result:
            task_id = state.get("id", str(a[0]))
            if statuses.get(task_id) != state["state"]:
                statuses[task_id] = state["state"]
                record(
                    "state",
                    task_id=task_id,
                    **{
                        k: v
                        for k, v in state.items()
                        if k
                        in (
                            "state",
                            "creation_timestamp_ms",
                            "start_timestamp_ms",
                            "update_timestamp_ms",
                            "batch_eecu_usage_seconds",
                            "error_message",
                        )
                    },
                )
        return result

    def timed_download(service, remote, path):
        sid = path.stem
        begin = time.perf_counter()
        record("download_start", sample_id=sid)
        result = original_download(service, remote, path)
        metrics.setdefault(sid, {}).update(
            download_s=time.perf_counter() - begin, bytes=int(remote["size"])
        )
        record("download_end", sample_id=sid)
        return result

    def timed_ingest(path, sample, names, output):
        nonlocal ingest_depth
        outer = ingest_depth == 0
        ingest_depth += 1
        begin = time.perf_counter()
        try:
            result = original_ingest(path, sample, names, output)
        finally:
            ingest_depth -= 1
        if outer:
            sid = sample["sample_id"]
            metrics.setdefault(sid, {})["ingest_s"] = time.perf_counter() - begin
            record("ingest_end", sample_id=sid)
        return result

    with (
        patch.object(gs, "drive_client", lambda unused: drive),
        patch.object(gs, "download", timed_download),
        patch.object(gs, "ingest", timed_ingest),
        patch.object(ee.data, "getTaskStatus", timed_status),
        patch.object(ee.batch.Task, "start", timed_start),
    ):
        gs.run(
            ee,
            SimpleNamespace(
                batch_size=32, concurrency=args.concurrency, poll_seconds=args.poll_seconds
            ),
            root,
            rows,
        )
    return {
        "wall_s": time.perf_counter() - started,
        "all_committed_s": max(e["elapsed_s"] for e in events if e["kind"] == "ingest_end"),
        "samples": [dict(sample_id=sid, **values) for sid, values in metrics.items()],
        "events": events,
    }


def run(args):
    rows = load_rows(args)
    root = args.output / args.command / args.block
    root.mkdir(parents=True, exist_ok=False)
    logging.getLogger().addHandler(logging.FileHandler(root / "run.log"))
    ee.Initialize(project=args.project, opt_url=args.endpoint)
    from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.data.normalize import load_computed_config

    load_computed_config()
    # Warm only local imports before timing; no remote sample query is warmed.
    import shapely  # noqa: F401
    import zarr  # noqa: F401

    started = datetime.now(UTC).isoformat()
    try:
        result = (run_drive if args.command == "drive" else run_direct)(args, rows, root)
    except Exception as exc:
        save(root / "error.json", {"error": str(exc), "started_at": started})
        raise
    result.update(
        method=args.command,
        block=args.block,
        concurrency=args.concurrency,
        poll_seconds=args.poll_seconds if args.command == "drive" else None,
        endpoint=args.endpoint,
        started_at=started,
        finished_at=datetime.now(UTC).isoformat(),
        count=len(rows),
        output=str(root),
    )
    save(root / "result.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("samples", "events")}, indent=2))


def pixels(path, sample, names):
    with rasterio.open(path) as src:
        expected = Affine(*sample["transform"])
        col, row = (~src.transform) * (expected.c, expected.f)
        window = Window(round(col), round(row), gs.SIZE, gs.SIZE)
        assert src.crs.to_string() == sample["crs"], path
        assert np.allclose(
            list(src.window_transform(window))[:6], sample["transform"], atol=1e-6, rtol=0
        ), path
        assert list(src.descriptions) == names, path
        data = src.read(window=window)
        assert data.shape == (len(names), gs.SIZE, gs.SIZE), path
        assert np.isfinite(data).all(), path
        return data


def compare(args):
    import zarr

    selection = json.loads((args.output / "selection.json").read_text())
    comparisons = []
    for item in selection["samples"]:
        sid, block = item["row"]["sample_id"], item["block"]
        direct = args.output / "direct" / block
        metadata_path = direct / "downloads" / f"{sid}.json"
        if not metadata_path.exists() or not (direct / f"timing_{sid}.json").exists():
            continue
        metadata = json.loads(metadata_path.read_text())
        sample, names = metadata["sample"], metadata["bands"]
        actual = pixels(direct / "downloads" / f"{sid}.tif", sample, names)
        for kind, reference in [
            ("fresh_drive", args.output / "drive" / block),
            ("archived_drive", Path(selection["source"])),
        ]:
            reference_path = reference / "downloads" / f"{sid}.tif"
            markers = [
                reference / "zarr" / sid / name / "_SUCCESS.json"
                for name in ("inputs.zarr", "labels.zarr")
            ]
            if not reference_path.exists() or not all(p.exists() for p in markers):
                continue
            expected = pixels(reference_path, sample, names)
            differences = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
            mask_indices = [i for i, name in enumerate(names) if name.endswith("_valid")]
            zarr_differences = []
            for store in ("inputs.zarr", "labels.zarr"):
                left = zarr.open_group(str(direct / "zarr" / sid / store), mode="r")
                right = zarr.open_group(str(reference / "zarr" / sid / store), mode="r")

                def check_array(name, obj):
                    if isinstance(obj, zarr.Array):
                        if name not in right or not np.array_equal(obj[:], right[name][:]):
                            zarr_differences.append(f"{store}/{name}")

                left.visititems(check_array)
            comparisons.append(
                dict(
                    sample_id=sid,
                    reference=kind,
                    pixel_exact=bool(np.array_equal(actual, expected)),
                    masks_exact=bool(np.array_equal(actual[mask_indices], expected[mask_indices])),
                    max_abs_error=float(differences.max()),
                    different_values=int(np.count_nonzero(differences)),
                    differing_zarr_arrays=zarr_differences,
                )
            )
    runs = [json.loads(p.read_text()) for p in args.output.glob("*/*/result.json")]

    def distribution(values):
        return {
            "count": len(values),
            "mean_s": float(np.mean(values)),
            "median_s": float(np.median(values)),
            "p90_s": float(np.percentile(values, 90)),
        }

    totals = {}
    for method in ("drive", "direct"):
        selected_runs = [r for r in runs if r["method"] == method]
        if not selected_runs:
            continue
        count = sum(r["count"] for r in selected_runs)
        wall = sum(r["wall_s"] for r in selected_runs)
        committed = sum(r["all_committed_s"] for r in selected_runs)
        metrics = [sample for r in selected_runs for sample in r["samples"]]
        stages = {}
        for key in (
            "download_s",
            "prep_s",
            "compute_and_receive_s",
            "write_fsync_checksum_s",
            "ingest_s",
            "total_s",
        ):
            values = [sample[key] for sample in metrics if key in sample]
            if values:
                stages[key] = distribution(values)
        completed = [
            e for r in selected_runs for e in r.get("events", []) if e.get("state") == "COMPLETED"
        ]
        if completed:
            stages["server_queue_s"] = distribution(
                [(e["start_timestamp_ms"] - e["creation_timestamp_ms"]) / 1000 for e in completed]
            )
            stages["server_execution_s"] = distribution(
                [(e["update_timestamp_ms"] - e["start_timestamp_ms"]) / 1000 for e in completed]
            )
        totals[method] = dict(
            count=count,
            wall_s=wall,
            all_committed_s=committed,
            wall_s_per_sample=wall / count,
            samples_per_hour=3600 * count / wall,
            stages=stages,
        )
    fresh = [r for r in comparisons if r["reference"] == "fresh_drive"]
    complete = len(fresh) == len(selection["samples"]) and all(
        totals.get(method, {}).get("count") == len(selection["samples"])
        for method in ("drive", "direct")
    )
    exact = all(
        r["pixel_exact"] and r["masks_exact"] and not r["differing_zarr_arrays"] for r in fresh
    )
    summary = {
        "complete": complete,
        "fresh_pairs_exact": exact if fresh else None,
        "timing_totals": totals,
        "runs": [{k: v for k, v in run.items() if k not in ("samples", "events")} for run in runs],
        "comparisons": comparisons,
    }
    if complete:
        summary["speedup_including_cleanup"] = (
            totals["drive"]["wall_s"] / totals["direct"]["wall_s"]
        )
        summary["speedup_until_local_commit"] = (
            totals["drive"]["all_committed_s"] / totals["direct"]["all_committed_s"]
        )
    save(args.output / "comparison.json", summary)
    print(json.dumps(summary, indent=2))
    if complete and not exact:
        raise ValueError("Paired outputs differ; inspect comparison.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "drive", "direct", "compare"))
    parser.add_argument(
        "--source", type=Path, default=Path("/mnt/d/Hydrostate/samples/batch_seed43_n10000")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project", default="cropland-477906")
    parser.add_argument("--drive-token", default="drive_token.json")
    parser.add_argument(
        "--drive-quota-project", help="Reuse EE OAuth scopes with this existing Drive API project"
    )
    parser.add_argument("--block", choices=("A", "B"), default="A")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--endpoint", default="https://earthengine.googleapis.com")
    args = parser.parse_args()
    if args.concurrency < 1 or args.poll_seconds < 1:
        parser.error("Concurrency and poll interval must be positive")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.command == "prepare":
        prepare(args)
    elif args.command == "compare":
        compare(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
