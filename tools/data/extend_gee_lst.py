#!/usr/bin/env python
"""Publish one LST observation per sample, reusing existing Landsat acquisitions."""

import argparse
import fcntl
import json
import logging
import threading
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from hydrostate.data.gee_lst import MAX_UNCERTAINTY_K, PRODUCER_VERSION, fetch_sample, variables
from hydrostate.datasets.zarr_dataset import _ShardCache
from hydrostate.storage.packs import VariablePack, VariablePackWriter, file_hash, validate_samples

if __package__:
    from .extend_gee_labels import Checkpoint, download, select_rows
else:
    from extend_gee_labels import Checkpoint, download, select_rows

LOG = logging.getLogger(__name__)


class Fetcher:
    def __init__(self, manifest, baseline):
        self.root = manifest.parent
        table = pd.read_parquet(manifest)
        self.baseline = VariablePack(baseline, table, kind="label", variables=["lst"])
        index = pd.read_parquet(baseline / "index.parquet")
        self.scenes = {r.sample_id: json.loads(r.provenance)["scenes"]["lst"]
                       for r in index.itertuples()}
        self.local = threading.local()

    def __call__(self, ee, row):
        if not hasattr(self.local, "cache"):
            self.local.cache = _ShardCache()
        cache = self.local.cache
        path = Path(row["input_shard"])
        group = cache.get(str(path if path.is_absolute() else self.root / path))
        index = int(row["input_row"])
        sensor_valid = np.asarray(group["sensor_valid"][index], bool)
        masks = {name: np.asarray(group[name + "_valid"][index], bool)
                 & sensor_valid[:, mi, None, None, None]
                 for mi, name in enumerate(("s1", "s2", "landsat"))}
        observed = np.stack([value.any(0) for value in masks.values()]).any(0)
        observed = observed.reshape(1, 40, 3, 40, 3).all(axis=(2, 4))
        paired = masks["landsat"].reshape(-1, 1, 40, 3, 40, 3).all(axis=(3, 5))
        previous = (self.baseline.read(row["sample_id"], cache), self.scenes.get(row["sample_id"]))
        return fetch_sample(ee, row, observed, paired, previous)


def publish(rows, cache, args, contract):
    counts = Counter()
    pixels = 0
    with VariablePackWriter(args.output, args.manifest, kind="label", variables=variables(),
                            source=contract, version=PRODUCER_VERSION, shard_size=256,
                            chunk_samples=16) as writer:
        for i, row in enumerate(rows):
            values, provenance = cache.read(row["sample_id"])
            valid = values["lst_valid"]
            if np.any(valid & (values["lst_uncertainty"] > MAX_UNCERTAINTY_K)):
                raise ValueError("Checkpoint contains LST above the fixed 2 K threshold")
            writer.write(row["sample_id"], values, provenance=provenance)
            counts[provenance["selection"]] += 1
            pixels += int(valid.sum())
            if (i + 1) % 256 == 0:
                LOG.info("Wrote LST Zarr %d/%d", i + 1, len(rows))
        if file_hash(args.manifest) != contract["manifest_sha256"]:
            raise RuntimeError("Manifest changed during export")
        if file_hash(args.base_pack / "metadata.json") != contract["base_pack_metadata_sha256"]:
            raise RuntimeError("Baseline pack changed during export")
        report = dict(samples=len(rows), samples_with_labels=len(rows) - counts["missing"],
                      valid_pixels_30m=pixels, selection_counts=dict(counts),
                      max_uncertainty_k=MAX_UNCERTAINTY_K, contract=contract)
        (writer.partial / "coverage.json").write_text(json.dumps(report, indent=2) + "\n")
        writer.finalize()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base-pack", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--project", default="cropland-477906")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--publish-only", action="store_true")
    args = parser.parse_args()
    if args.limit < 0 or not 1 <= args.workers <= 32 or args.attempts < 1:
        parser.error("Invalid limit, workers or attempts")
    args.manifest = args.manifest.resolve()
    args.base_pack = args.base_pack.resolve()
    if args.output.exists():
        raise FileExistsError(f"Refusing to replace published data: {args.output}")
    args.work_dir = args.work_dir or args.output.with_name("." + args.output.name + ".work")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(args.work_dir / "run.log")])
    table = pd.read_parquet(args.manifest)
    validate_samples(table)
    rows = select_rows(table, args.limit, args.seed)
    fetcher = Fetcher(args.manifest, args.base_pack)
    contract = dict(producer=PRODUCER_VERSION, manifest_sha256=file_hash(args.manifest),
                    base_pack_metadata_sha256=file_hash(args.base_pack / "metadata.json"),
                    max_uncertainty_k=MAX_UNCERTAINTY_K,
                    policy="preserve_good_same_day;try_same_day;nearest_existing_Landsat;one_actual_time")
    with (args.work_dir / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cache = Checkpoint(args.work_dir / "checkpoint.sqlite3", contract)
        try:
            LOG.info("Selected %d/%d samples; output=%s", len(rows), len(table), args.output)
            if not args.publish_only:
                download(rows, cache, args, fetcher)
            print(json.dumps(publish(rows, cache, args, contract), indent=2), flush=True)
        finally:
            cache.close()


if __name__ == "__main__":
    main()
