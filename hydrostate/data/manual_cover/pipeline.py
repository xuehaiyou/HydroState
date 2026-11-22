"""Resumable label conversion, bounded input collection, and publication gates."""

import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from numcodecs import Blosc

from hydrostate.data.cover_points.contract import config_identity
from hydrostate.data.cover_points.prepare import atomic_parquet
from hydrostate.data.gee.contracts import atomic_json, fingerprint, locked
from hydrostate.data.retrieval.alignment import grid_metadata, sensor_masks
from hydrostate.data.retrieval.collector import input_arrays
from hydrostate.storage import SampleShardWriter
from hydrostate.storage.packs import file_hash

from .contract import VERSION, ManualCoverSchema, producer_identity
from .labels import area_fractions, event_regions, point_fractions


def configuration(root):
    root = Path(root)
    config = json.loads((root / "collection.json").read_text())
    if config["producer_identity"] != producer_identity():
        raise ValueError("Frozen producer changed; use its original Git release")
    if (root / "_PUBLISHED.json").exists():
        raise ValueError("Published release is immutable")
    return root, config, pd.read_parquet(root / "plan.parquet")


def selected_pilot(plan):
    # Each source region and each split is represented before full acquisition.
    return plan.groupby("region", sort=True).head(1).sample_id.tolist()


def labels(root):
    root, config, plan = configuration(root)
    groups = {k: v for k, v in pd.read_parquet(root / "points.parquet").groupby("sample_id")}
    started = time.monotonic()
    with locked(root / ".labels.lock", blocking=False):
        receipts = root / "label_records"
        receipts.mkdir(exist_ok=True)
        for i, row in enumerate(
            plan.sort_values(["source", "event", "sample_id"]).to_dict("records")
        ):
            sid = row["sample_id"]
            record = receipts / f"{sid}.json"
            if record.exists():
                continue
            if row["source"] == "glacier":
                arrays = point_fractions(groups[sid])
            else:
                water, valid = event_regions(str(root), row["event"], row["crs"])
                arrays = area_fractions(water, valid, json.loads(row["transform"]))
            path = root / "labels" / f"{sid}.zarr"
            if path.exists():
                shutil.rmtree(path)  # no receipt: interrupted, uncommitted output only
            path.parent.mkdir(exist_ok=True)
            z = zarr.open_group(str(path), mode="w")
            for name, values in arrays.items():
                z.create_dataset(
                    name,
                    data=values[None],
                    chunks=(1, *values.shape),
                    compressor=Blosc(cname="zstd", clevel=3),
                )
            z.attrs.update(
                dict(
                    schema_version=VERSION,
                    config_sha256=config_identity(config),
                    source=row["source"],
                    num_samples=1,
                    units="1",
                    label_support=config["point_assumption"]
                    if row["source"] == "glacier"
                    else config["flood_geometry"],
                )
            )
            atomic_json(path / "_SUCCESS.json", dict(complete=True, num_samples=1))
            counts = {c: int(arrays[c + "_valid"].sum()) for c in ("water", "snow_ice", "land")}
            atomic_json(
                record,
                dict(
                    sample_id=sid,
                    label_path=f"labels/{sid}.zarr",
                    valid_counts=counts,
                    conflict_cells=int(arrays["point_conflict"].sum()),
                    water_sum=float(arrays["water"][arrays["water_valid"].astype(bool)].sum()),
                ),
            )
            if i % 100 == 0:
                print(
                    json.dumps(dict(stage="labels", processed=i + 1, total=len(plan))), flush=True
                )
        records = pd.DataFrame([json.loads(p.read_text()) for p in sorted(receipts.glob("*.json"))])
        table = plan.merge(records, on="sample_id", validate="one_to_one")
        table["label_usable"] = table.valid_counts.map(lambda x: x["water"] > 0)
        atomic_parquet(table, root / "label_inventory.parquet")
        atomic_json(
            root / "_LABELS_READY.json",
            dict(
                complete=True,
                windows=len(table),
                usable_windows=int(table.label_usable.sum()),
                seconds=time.monotonic() - started,
                inventory_sha256=file_hash(root / "label_inventory.parquet"),
            ),
        )
    return dict(windows=len(table), usable_windows=int(table.label_usable.sum()))


def collect(root, project, pilot=False, concurrency=4):
    if not 1 <= concurrency <= 4:
        raise ValueError("Use 1 to 4 concurrent collectors")
    root, config, _ = configuration(root)
    inventory = pd.read_parquet(root / "label_inventory.parquet")
    if not (root / "_LABELS_READY.json").exists():
        raise ValueError("Finish labels first")
    if not pilot and not (root / "audit/pilot.json").exists():
        raise ValueError("Representative input pilot must pass before expansion")
    if not pilot and not json.loads((root / "audit/pilot.json").read_text())["passed"]:
        raise ValueError("Pilot failed")
    from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.data.normalize import load_computed_config

    from hydrostate.data.gee.client import initialize

    ee, norm = initialize(project), load_computed_config()
    schema = ManualCoverSchema()
    glacier_root = Path(config["glacier_root"])
    old = pd.read_parquet(glacier_root / "samples.parquet").set_index("sample_id")
    old_config = json.loads((glacier_root / "collection.json").read_text())
    with locked(root / ".collection.lock", blocking=False):
        for name in ("records", "rejected", "cache"):
            (root / name).mkdir(exist_ok=True)
        selected = (
            inventory[inventory.sample_id.isin(selected_pilot(inventory[inventory.label_usable]))]
            if pilot
            else inventory
        )

        def process(row):
            sid = row["sample_id"]
            receipt, rejected = root / "records" / f"{sid}.json", root / "rejected" / f"{sid}.json"
            if receipt.exists() or rejected.exists():
                return
            if not row["label_usable"]:
                atomic_json(rejected, dict(sample_id=sid, reason="no_full_valid_label_pixel"))
                return
            z = zarr.open_group(str(root / row["label_path"]), mode="r")
            targets = {n: z[n][0] for n in schema.targets}
            if row["source"] == "glacier":
                prior = old.loc[row["source_sample_id"]]
                atomic_json(
                    receipt,
                    dict(
                        row,
                        input_path=str(glacier_root / prior.input_path),
                        input_schema_version=old_config["version"],
                        input_config_sha256=config_identity(old_config),
                        shard_index=0,
                        input_reused=True,
                    ),
                )
                return
            sample = dict(
                row, transform=json.loads(row["transform"]), block=json.loads(row["block"])
            )
            arrays, scenes, eligible = input_arrays(ee, sample, root / "cache", norm, config)
            observed = np.logical_or.reduce(
                [v.any(axis=0)[0] for v in sensor_masks(arrays).values()]
            )
            support = observed.reshape(40, 3, 40, 3).all(axis=(1, 3))
            if not eligible or not np.any(support & targets["water_valid"][0].astype(bool)):
                atomic_json(rejected, dict(sample_id=sid, reason="input_coverage"))
                return
            base = root / "zarr" / sid
            if base.exists():
                shutil.rmtree(base)
            writer = SampleShardWriter(
                str(base / "inputs.zarr"),
                str(base / "labels.zarr"),
                1,
                schema,
                attrs=dict(
                    config_sha256=config_identity(config),
                    input_profile=config["input_profile"],
                    input_grids=grid_metadata(sample),
                    scenes=scenes,
                    normalization_hash=fingerprint(norm),
                ),
            )
            writer.write(arrays, targets)
            writer.finalize()
            atomic_json(
                receipt,
                dict(
                    row,
                    input_path=f"zarr/{sid}/inputs.zarr",
                    input_schema_version=VERSION,
                    input_config_sha256=config_identity(config),
                    shard_index=0,
                    input_reused=False,
                ),
            )

        pending = selected.to_dict("records")
        try:
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                for i, _ in enumerate(pool.map(process, pending)):
                    print(
                        json.dumps(dict(stage="inputs", processed=i + 1, total=len(pending))),
                        flush=True,
                    )
        finally:
            records = [json.loads(p.read_text()) for p in sorted((root / "records").glob("*.json"))]
            if records:
                atomic_parquet(pd.DataFrame(records), root / "samples.parquet")
            atomic_json(
                root / "collection_report.json",
                dict(
                    accepted=len(records),
                    rejected=len(list((root / "rejected").glob("*.json"))),
                    planned=len(inventory),
                    pilot=pilot,
                ),
            )
    return json.loads((root / "collection_report.json").read_text())
