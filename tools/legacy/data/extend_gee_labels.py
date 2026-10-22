#!/usr/bin/env python
"""Resume direct GEE downloads and publish ecosystem labels for existing samples."""

from __future__ import annotations

import argparse
import fcntl
import io
import json
import logging
import sqlite3
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from hydrostate.data.gee_labels import (
    COLLECTIONS,
    PRODUCER_VERSION,
    fetch_sample,
    variables,
)
from hydrostate.storage import VariablePackWriter
from hydrostate.storage.packs import file_hash, validate_samples

LOG = logging.getLogger(__name__)


def select_rows(table, limit=0, seed=43):
    """Deterministic pilot covering years, seasons, cover bins, climate and latitude."""
    rows = table.sort_values("sample_id").to_dict("records")
    if not limit or limit >= len(rows):
        return rows
    rng = np.random.default_rng(seed)
    rng.shuffle(rows)
    features = []
    for row in rows:
        day = str(row["target_time"])
        features.append({("year", day[:4]), ("quarter", (int(day[5:7]) - 1) // 3),
                         ("latitude", int(row["lat"] // 20)), ("split", row["split"]),
                         ("cover", row.get("frequency_bin")),
                         ("climate", row.get("climate_class"))})
    chosen, seen, available = [], set(), set(range(len(rows)))
    for _ in range(limit):
        index = max(sorted(available), key=lambda i: len(features[i] - seen))
        seen.update(features[index])
        chosen.append(rows[index])
        available.remove(index)
    return chosen


class Checkpoint:
    """A single transactional cache; no per-sample scratch files or false successes."""

    def __init__(self, path, contract):
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS config (value TEXT NOT NULL)")
        self.db.execute("CREATE TABLE IF NOT EXISTS samples "
                        "(id TEXT PRIMARY KEY, payload BLOB NOT NULL, provenance TEXT NOT NULL)")
        self.db.execute("CREATE TABLE IF NOT EXISTS errors (id TEXT PRIMARY KEY, message TEXT)")
        encoded = json.dumps(contract, sort_keys=True)
        previous = self.db.execute("SELECT value FROM config").fetchone()
        if previous and previous[0] != encoded:
            self.db.close()
            raise ValueError("Checkpoint parameters/manifest differ; use a new work directory")
        if not previous:
            self.db.execute("INSERT INTO config VALUES (?)", (encoded,))
            self.db.commit()

    def completed(self):
        return {row[0] for row in self.db.execute("SELECT id FROM samples")}

    def save(self, sid, values, provenance):
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **values)
        with self.db:
            self.db.execute("INSERT INTO samples VALUES (?, ?, ?)",
                            (sid, buffer.getvalue(), json.dumps(provenance, allow_nan=False)))
            self.db.execute("DELETE FROM errors WHERE id = ?", (sid,))

    def fail(self, sid, error):
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO errors VALUES (?, ?)", (sid, str(error)))

    def read(self, sid):
        row = self.db.execute("SELECT payload, provenance FROM samples WHERE id = ?",
                              (sid,)).fetchone()
        if row is None:
            raise ValueError(f"Sample {sid} has not completed")
        with np.load(io.BytesIO(row[0]), allow_pickle=False) as archive:
            values = {name: archive[name] for name in archive.files}
        return values, json.loads(row[1])

    def close(self):
        self.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        self.db.close()


def fetch_with_retry(ee, row, args, fetcher=None):
    for attempt in range(args.attempts):
        try:
            return (fetcher(ee, row) if fetcher is not None
                    else fetch_sample(ee, row, capacity=args.native_capacity))
        except (ValueError, TypeError):
            raise  # Invalid data/contracts need correction, not network retries.
        except Exception:
            if attempt + 1 == args.attempts:
                raise
            time.sleep(min(2 ** attempt, 8))


def download(rows, cache, args, fetcher=None):
    done = cache.completed()
    remaining = [row for row in rows if row["sample_id"] not in done]
    if not remaining:
        return
    import ee

    ee.Initialize(project=args.project)
    ee.data.setDeadline(60000)
    started = time.perf_counter()
    failed, completed = 0, len(rows) - len(remaining)
    iterator = iter(remaining)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {}

        def fill():
            while len(pending) < args.workers * 2:
                row = next(iterator, None)
                if row is None:
                    break
                pending[pool.submit(fetch_with_retry, ee, row, args, fetcher)] = row["sample_id"]

        fill()
        last_report = 0
        while pending:
            ready, _ = wait(pending, timeout=20, return_when=FIRST_COMPLETED)
            for future in ready:
                sid = pending.pop(future)
                try:
                    values, provenance = future.result()
                    cache.save(sid, values, provenance)
                    completed += 1
                except Exception as exc:
                    cache.fail(sid, exc)
                    failed += 1
                    LOG.error("Failed %s: %s", sid, exc)
            fill()
            elapsed = time.perf_counter() - started
            if elapsed - last_report >= 20 or not pending:
                LOG.info("Downloaded %d/%d; failed=%d; elapsed=%.1fs",
                         completed, len(rows), failed, elapsed)
                last_report = elapsed
    if failed:
        raise RuntimeError(f"{failed} requests failed; successes saved; rerun to retry")


def publish(rows, cache, args, contract):
    counts = {name: dict(samples_with_labels=0, valid_observations=0) for name in variables()}
    report = dict(samples=len(rows), source_manifest=str(args.manifest.resolve()),
                  created_utc=datetime.now(UTC).isoformat(), contract=contract, variables=counts,
                  lst_same_day_scene_samples=0,
                  lst_uncertainty_thresholds={str(k): dict(samples=0, pixels_30m=0)
                                              for k in (2, 3, 5)})
    with VariablePackWriter(
        args.output, args.manifest, kind="label", source=contract, version=PRODUCER_VERSION,
        variables=variables(args.native_capacity), shard_size=256, chunk_samples=16,
    ) as writer:
        for i, row in enumerate(rows):
            values, provenance = cache.read(row["sample_id"])
            writer.write(row["sample_id"], values, provenance=provenance)
            report["lst_same_day_scene_samples"] += provenance["scenes"]["lst"] is not None
            for threshold, summary in report["lst_uncertainty_thresholds"].items():
                mask = values["lst_valid"] & (values["lst_uncertainty"] <= float(threshold))
                summary["samples"] += int(mask.any())
                summary["pixels_30m"] += int(mask.sum())
            for name in counts:
                mask = values[name + "_valid"]
                counts[name]["samples_with_labels"] += int(mask.any())
                counts[name]["valid_observations"] += int(mask.sum())
            if (i + 1) % 256 == 0:
                LOG.info("Wrote Zarr %d/%d", i + 1, len(rows))
        if file_hash(args.manifest) != contract["manifest_sha256"]:
            raise RuntimeError("Manifest changed during export")
        (writer.partial / "coverage.json").write_text(json.dumps(report, indent=2) + "\n")
        writer.finalize()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="New immutable label-pack directory")
    parser.add_argument("--work-dir", type=Path, help="Reusable SQLite checkpoint directory")
    parser.add_argument("--project", default="cropland-477906")
    parser.add_argument("--limit", type=int, default=0,
                        help="0 = all samples; >0 = representative pilot")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--native-capacity", type=int, default=64)
    parser.add_argument("--publish-only", action="store_true")
    args = parser.parse_args()
    if (args.limit < 0 or not 1 <= args.workers <= 32 or args.attempts < 1
            or args.native_capacity < 1):
        parser.error("Invalid limit, workers, attempts or capacity")
    if args.output.exists():
        raise FileExistsError(f"Refusing to replace published data: {args.output}")
    args.work_dir = args.work_dir or args.output.with_name("." + args.output.name + ".work")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(args.work_dir / "run.log")])
    table = pd.read_parquet(args.manifest)
    validate_samples(table)
    rows = select_rows(table, args.limit, args.seed)
    contract = dict(producer=PRODUCER_VERSION, manifest_sha256=file_hash(args.manifest),
                    collections=COLLECTIONS, native_capacity=args.native_capacity,
                    policy="same_DW;containing_LAI_GPP;same_day_LST_with_uncertainty;"
                           "full_BRDF;complete_native_cells")
    with (args.work_dir / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cache = Checkpoint(args.work_dir / "checkpoint.sqlite3", contract)
        try:
            LOG.info("Selected %d/%d existing samples; output=%s",
                     len(rows), len(table), args.output)
            if not args.publish_only:
                download(rows, cache, args)
            report = publish(rows, cache, args, contract)
            LOG.info("Published %s", args.output)
            print(json.dumps(report, indent=2), flush=True)
        finally:
            cache.close()


if __name__ == "__main__":
    main()
