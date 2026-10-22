"""Live Drive-versus-pixel parity check on one existing cached chunk per sensor."""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from hydrostate.data.gee.backends.drive import DriveTransport
from hydrostate.data.gee.backends.pixels import compute_pixels
from hydrostate.data.gee.client import initialize
from hydrostate.data.gee.contracts import atomic_json, fingerprint, input_shape, input_transform
from hydrostate.data.gee.products import BANDS, normalize_pixels, scene_expression
from hydrostate.data.gee.state import ExportState
from hydrostate.data.gee.workflows.drive_queue import run_jobs
from hydrostate.storage.scene_cache import SceneCache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, required=True, help="Existing regional cache (read-only)"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Separate disposable validation root"
    )
    parser.add_argument(
        "--download-config",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "configs/experiments/legacy/data/gee_drive.json",
    )
    args = parser.parse_args()
    if args.source.resolve() == args.output.resolve():
        parser.error("Validation output must differ from the production root")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.output.mkdir(parents=True, exist_ok=True)
    plan = json.loads((args.source / "plan.json").read_text())
    atomic_json(args.output / "plan.json", plan)
    cache = SceneCache(args.output, plan)
    settings = json.loads(args.download_config.read_text())
    jobs = []
    h, w = input_shape(cache.grid)
    tr = input_transform(cache.grid)
    for key in BANDS:
        chosen = None
        for path in sorted((args.source / "input_cache" / key).glob("*.zarr")):
            group = zarr.open_group(str(path), mode="r")
            if group.attrs["contract"] != cache.contract:
                raise ValueError("Existing cache contract differs")
            for cy, cx in np.argwhere(group["done"][:]):
                y, x = int(cy) * cache.chunk, int(cx) * cache.chunk
                if group["valid"][y : y + cache.chunk, x : x + cache.chunk].any():
                    chosen = (group.attrs["scene"], int(cy), int(cx))
                    break
            if chosen:
                break
        if not chosen:
            raise RuntimeError(f"No existing valid {key} chunk available for parity check")
        scene, cy, cx = chosen
        y, x = cy * cache.chunk, cx * cache.chunk
        width, height = min(cache.chunk, w - x), min(cache.chunk, h - y)
        job = dict(
            scene=scene,
            contract=cache.contract,
            grid=cache.grid,
            chunk=cache.chunk,
            bands=[*BANDS[key], "valid"],
            chunks=[[cy, cx]],
            x=x,
            y=y,
            width=width,
            height=height,
            transform=[10, 0, tr.c + x * 10, 0, -10, tr.f - y * 10],
            raw_bytes=width * height * (len(BANDS[key]) + 1) * 4,
            version=1,
        )
        job["id"] = fingerprint(job)[:24]
        jobs.append(job)
    ee = initialize(plan["config"]["gee_project"])
    transport = DriveTransport(
        ee, args.output, settings, quota_project=plan["config"]["gee_project"]
    )
    state = ExportState(args.output / "exports.sqlite")
    started = time.monotonic()
    try:
        run_jobs(transport, state, cache, jobs, settings)
    finally:
        state.close()
    drive_seconds = time.monotonic() - started
    results = []
    for job in jobs:
        start = time.monotonic()
        raw = compute_pixels(
            ee,
            scene_expression(ee, job["scene"]),
            job["bands"],
            cache.grid["crs"],
            job["transform"],
            job["width"],
            job["height"],
        )
        direct_seconds = time.monotonic() - start
        valid = raw["valid"] > 0
        expected = normalize_pixels(
            job["scene"]["key"], np.stack([raw[n] for n in job["bands"][:-1]]), valid, cache.norm
        )
        group = cache._open(job["scene"])
        x, y = job["x"], job["y"]
        actual = group["values"][:, y : y + job["height"], x : x + job["width"]]
        actual_valid = group["valid"][y : y + job["height"], x : x + job["width"]].astype(bool)
        np.testing.assert_array_equal(actual_valid, valid)
        np.testing.assert_array_equal(actual, expected)
        results.append(
            dict(
                sensor=job["scene"]["key"],
                asset=job["scene"]["asset_id"],
                valid_pixels=int(valid.sum()),
                pixel_seconds=direct_seconds,
                max_abs_difference=float(np.abs(actual.astype("f4") - expected).max()),
            )
        )
    atomic_json(
        args.output / "validation.json",
        dict(passed=True, drive_seconds=drive_seconds, checks=results),
    )
    print(json.dumps(dict(passed=True, drive_seconds=drive_seconds, checks=results), indent=2))


if __name__ == "__main__":
    main()
