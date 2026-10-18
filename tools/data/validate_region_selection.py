"""Compare fresh byte-mask selection with saved full-band regional selections."""

import argparse
import json
import logging
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import numpy as np
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hydrostate.data.gee.backends.pixels import PixelQualityCache
from hydrostate.data.gee.client import initialize
from hydrostate.data.gee.contracts import atomic_json
from hydrostate.data.gee.products import BANDS, slots
from hydrostate.data.gee.region_plan import export_jobs, selected_chunks, selected_export_jobs
from hydrostate.data.gee.selection import epoch, select_window
from hydrostate.data.gee.workflows.drive_queue import validate_settings
from hydrostate.storage.scene_cache import SceneCache

LOG = logging.getLogger(__name__)


def validate(source, output, *, limit=8, workers=4, download_config=None):
    if source.resolve() == output.resolve() or source.resolve() in output.resolve().parents:
        raise ValueError("Use a separate validation root outside the production root")
    if limit < 1 or workers < 1:
        raise ValueError("limit and workers must be positive")
    output.mkdir(parents=True, exist_ok=True)
    plan = json.loads((source / "plan.json").read_text())
    period = plan["periods"][0]
    with sqlite3.connect(f"file:{source / 'preparation.sqlite'}?mode=ro", uri=True) as db:
        saved = [
            json.loads(row[0])
            for row in db.execute(
                "SELECT result FROM windows WHERE date=? ORDER BY id", (period["time"],)
            )
        ]
    if not saved:
        raise ValueError("No saved full-band window selections to compare")
    indices = np.linspace(0, len(saved) - 1, min(limit, len(saved)), dtype=int)
    expected = [saved[i] for i in indices]
    first, last = slots(period["time"])[0][0], slots(period["time"])[-1][1]
    years = range(date.fromisoformat(first).year, date.fromisoformat(last).year + 1)
    scenes = {
        key: sorted(
            [
                s
                for year in years
                for s in json.loads((source / "catalogs" / f"{key}_{year}.json").read_text())
                if epoch(first) <= s["time"] < epoch(last)
            ],
            key=lambda s: (s["time"], s["asset_id"]),
            reverse=True,
        )
        for key in BANDS
    }
    # An empty full cache forces fresh mask requests, independent of the reference.
    full = SceneCache(output, plan)
    config_path = (
        download_config or Path(__file__).resolve().parents[2] / "configs/data/gee_drive.json"
    )
    settings = json.loads(config_path.read_text())
    validate_settings(settings, full)
    if (output / "input_cache").exists():
        raise ValueError("Validation root must not contain full-band inputs")
    quality = PixelQualityCache(full, ee=initialize(plan["config"]["gee_project"]))
    started = time.monotonic()

    def check(row):
        window = {k: row[k] for k in ("window_id", "row", "col")}
        actual = select_window(plan, period, window, scenes, quality)
        if actual != row:
            atomic_json(
                output / f"mismatch_{row['window_id']}.json", dict(expected=row, actual=actual)
            )
            raise AssertionError(f"Selection changed: {row['window_id']}")
        LOG.info("Exact selection parity: %s accepted=%s", row["window_id"], row["accepted"])
        return actual

    with ThreadPoolExecutor(workers) as pool:
        actual = list(pool.map(check, expected))
    selection_seconds = time.monotonic() - started
    # Also compare a valid 480-pixel chunk from each sensor against existing full data.
    checks = []
    for key in BANDS:
        found = None
        for path in sorted((source / "input_cache" / key).glob("*.zarr")):
            group = zarr.open_group(str(path), mode="r")
            if group.attrs["contract"] != full.contract:
                raise ValueError("Reference cache contract differs")
            for cy, cx in np.argwhere(group["done"][:]):
                y, x = int(cy) * full.chunk, int(cx) * full.chunk
                valid = group["valid"][y : y + full.chunk, x : x + full.chunk].astype(bool)
                if valid.any():
                    found = (group.attrs["scene"], y, x, valid)
                    break
            if found:
                break
        if found is None:
            raise ValueError(f"No valid full-band reference for {key}")
        scene, y, x, valid = found
        tile = quality.get_tile(scene, y // quality.tile, x // quality.tile)
        ys, xs = y % quality.tile, x % quality.tile
        np.testing.assert_array_equal(
            tile[ys : ys + valid.shape[0], xs : xs + valid.shape[1]], valid
        )
        checks.append(
            dict(
                sensor=key,
                asset_id=scene["asset_id"],
                pixels=int(valid.size),
                valid_pixels=int(valid.sum()),
            )
        )
        LOG.info("Exact full-chunk mask parity: %s %s", key, valid.shape)
    candidates = [s for group in scenes.values() for s in group]
    budget = settings["max_export_gib"] * 1024**3
    old_jobs = list(export_jobs(full, candidates, actual, max_export_bytes=budget))
    new_jobs = list(
        selected_export_jobs(
            full,
            selected_chunks(full, actual),
            max_export_bytes=budget,
            max_padding_fraction=settings.get("max_padding_fraction", 0.25),
        )
    )
    report = dict(
        passed=True,
        download_settings=settings,
        windows=len(actual),
        time=period["time"],
        selection_seconds=selection_seconds,
        checks=checks,
        mask_requests=quality.requests,
        mask_response_bytes=quality.response_bytes,
        cold_subset_candidate_exports=len(old_jobs),
        cold_subset_candidate_bytes=sum(j["raw_bytes"] for j in old_jobs),
        cold_subset_selected_exports=len(new_jobs),
        cold_subset_selected_bytes=sum(j["raw_bytes"] for j in new_jobs),
        note="Subset byte planning comparison, not a full-region throughput benchmark",
    )
    atomic_json(output / "validation.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--download-config", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(
        json.dumps(
            validate(
                args.source,
                args.output,
                limit=args.limit,
                workers=args.workers,
                download_config=args.download_config,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
