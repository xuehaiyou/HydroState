"""Small live native-versus-aligned parity check; production input cache is read-only."""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
import zarr
from affine import Affine

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from hydrostate.data.gee.backends.drive import DriveTransport
from hydrostate.data.gee.backends.pixels import compute_pixels
from hydrostate.data.gee.client import initialize
from hydrostate.data.gee.contracts import atomic_json
from hydrostate.data.gee.native import NativeReader, native_expression, native_job, source_groups
from hydrostate.data.gee.products import BANDS, normalize_pixels, scene_expression
from hydrostate.data.gee.region_plan import selected_export_jobs
from hydrostate.data.gee.state import ExportState
from hydrostate.data.gee.workflows.drive_queue import run_jobs
from hydrostate.storage.scene_cache import SceneCache


def choose(source, cache, count):
    jobs = []
    for key in BANDS:
        selected = []
        for path in sorted((source / "input_cache" / key).glob("*.zarr")):
            group = zarr.open_group(str(path), mode="r")
            if group.attrs["contract"] != cache.contract:
                raise ValueError("Reference cache contract differs")
            # Include cloud/edge pixels, without benchmarking a wholly empty image.
            for cy, cx in np.argwhere(group["done"][:]):
                y, x = int(cy)*cache.chunk, int(cx)*cache.chunk
                fraction = group["valid"][y:y+cache.chunk, x:x+cache.chunk].mean()
                if 0 < fraction < 1:
                    selected.append((group.attrs["scene"], (int(cy), int(cx))))
                    break
            if len(selected) == count:
                break
        if len(selected) < count:
            raise ValueError(f"Need {count} cached {key} scenes with valid and invalid pixels")
        for scene, cell in selected:
            jobs.extend(selected_export_jobs(
                cache, [dict(scene=scene, chunks={cell})], max_export_bytes=1e8
            ))
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunks-per-sensor", type=int, default=2)
    parser.add_argument("--drive-sensor", choices=list(BANDS),
                        help="Also validate one actual multipart Drive export for this sensor")
    args = parser.parse_args()
    if args.source.resolve() == args.output.resolve() or args.chunks_per_sensor < 1:
        parser.error("Use a separate validation output and a positive sample count")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.output.mkdir(parents=True, exist_ok=True)
    plan = json.loads((args.source / "plan.json").read_text())
    cache = SceneCache(args.output, plan)
    reference = SceneCache(args.source, plan)
    ee = initialize(plan["config"]["gee_project"])
    jobs_path = args.output / "jobs.json"
    if jobs_path.exists():
        jobs = json.loads(jobs_path.read_text())
    else:
        jobs = [
            native_job(j, source_groups(ee, args.output, j["scene"]))
            for j in choose(args.source, cache, args.chunks_per_sensor)
        ]
        atomic_json(jobs_path, jobs)
    checks = []
    for job in jobs:
        key = job["scene"]["key"]
        logging.info("Validating %s %s", key, job["scene"]["asset_id"])
        paths = {}
        started = time.monotonic()
        for part in job["parts"]:
            path = args.output / (part["id"] + ".tif")
            raw = compute_pixels(ee, native_expression(ee, part), part["bands"], part["crs"],
                                 part["transform"], part["width"], part["height"])
            with rasterio.open(
                path, "w", driver="GTiff", width=part["width"], height=part["height"],
                count=len(part["bands"]), dtype=part["dtype"], crs=part["crs"],
                transform=Affine(*part["transform"]),
            ) as dst:
                dst.write(np.stack([raw[b] for b in part["bands"]]))
                dst.descriptions = tuple(part["bands"])
            paths[part["id"]] = [path]
        transfer_seconds = time.monotonic() - started
        started = time.monotonic()
        with NativeReader(job, paths) as reader:
            values, valid = reader.read(job["transform"], job["width"], job["height"])
        aligned = normalize_pixels(key, values, valid, cache.norm)
        local_seconds = time.monotonic() - started
        started = time.monotonic()
        baseline = compute_pixels(ee, scene_expression(ee, job["scene"]), job["bands"],
                                  job["grid"]["crs"], job["transform"],
                                  job["width"], job["height"])
        baseline_seconds = time.monotonic() - started
        expected_valid = baseline["valid"] > 0
        expected = normalize_pixels(
            key, np.stack([baseline[b] for b in BANDS[key]]), expected_valid, cache.norm
        )
        group = reference._open(job["scene"])
        y, x, h, w = job["y"], job["x"], job["height"], job["width"]
        archived = group["values"][:, y:y+h, x:x+w]
        archived_valid = group["valid"][y:y+h, x:x+w].astype(bool)
        check = dict(
            sensor=key, asset=job["scene"]["asset_id"], chunks=job["chunks"],
            masks_exact=bool(np.array_equal(valid, expected_valid)),
            normalized_exact=bool(np.array_equal(aligned, expected)),
            archived_exact=bool(np.array_equal(aligned, archived)
                                and np.array_equal(valid, archived_valid)),
            differing_masks=int(np.count_nonzero(valid != expected_valid)),
            differing_values=int(np.count_nonzero(aligned != expected)),
            max_abs_difference=float(np.abs(aligned.astype("f4") - expected).max()),
            valid_pixels=int(valid.sum()), native_raw_bytes=job["raw_bytes"],
            aligned_raw_bytes=job["legacy_raw_bytes"],
            native_requests=len(job["parts"]), native_seconds=transfer_seconds,
            local_seconds=local_seconds, aligned_seconds=baseline_seconds,
        )
        checks.append(check)
        atomic_json(args.output / "validation.json", dict(checks=checks, passed=False))
        print(json.dumps(check), flush=True)
    passed = all(c["masks_exact"] and c["normalized_exact"] and c["archived_exact"] for c in checks)
    result = dict(passed=passed, checks=checks, drive=None)
    atomic_json(args.output / "validation.json", result)
    if not passed:
        raise ValueError("Native input parity failed; do not enable this layout on production")
    if args.drive_sensor:
        job = next(j for j in jobs if j["scene"]["key"] == args.drive_sensor)
        settings = json.loads(Path("configs/experiments/legacy/data/gee_drive_native.json").read_text())
        transport = DriveTransport(ee, args.output, settings,
                                   quota_project=plan["config"]["gee_project"])
        state = ExportState(args.output / "exports.sqlite")
        started = time.monotonic()
        try:
            run_jobs(transport, state, cache, [job], settings)
        finally:
            state.close()
        src, dst = reference._open(job["scene"]), cache._open(job["scene"])
        y, x, h, w = job["y"], job["x"], job["height"], job["width"]
        np.testing.assert_array_equal(src["values"][:, y:y+h, x:x+w],
                                      dst["values"][:, y:y+h, x:x+w])
        np.testing.assert_array_equal(src["valid"][y:y+h, x:x+w], dst["valid"][y:y+h, x:x+w])
        result["drive"] = dict(passed=True, sensor=args.drive_sensor,
                               seconds=time.monotonic()-started, job_id=job["id"])
        atomic_json(args.output / "validation.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
