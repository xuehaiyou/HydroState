"""Bounded, resumable acquisition of unchanged retrieval inputs for human points."""

import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from hydrostate.data.gee.contracts import atomic_json, fingerprint, locked
from hydrostate.data.retrieval.alignment import grid_metadata, sensor_masks
from hydrostate.data.retrieval.collector import input_arrays
from hydrostate.storage import SampleShardWriter

from .contract import CoverPointSchema, config_identity, producer_identity
from .prepare import atomic_parquet


def point_targets(points):
    return dict(
        cover_point_class=points.cover_class.to_numpy("int8"),
        cover_point_original_class=points.original_class.to_numpy("int8"),
        cover_point_valid=np.ones(len(points), "uint8"),
        cover_point_rc=points[["output_row", "output_col"]].to_numpy("int16"),
        cover_point_lonlat=points[["lon", "lat"]].to_numpy("float64"),
        cover_point_date=(
            pd.to_datetime(points.observation_date, utc=True).map(lambda t: int(t.timestamp()))
        ).to_numpy("int64"),
        cover_point_index=points.point_index.to_numpy("int64"),
    )


def collect_one(ee, row, points, root, config, norm):
    sid = row["sample_id"]
    committed, rejection = root / "records" / f"{sid}.json", root / "rejected" / f"{sid}.json"
    if committed.exists() or rejection.exists():
        return sid
    sample = dict(row, transform=json.loads(row["transform"]), block=json.loads(row["block"]))
    arrays, scenes, eligible = input_arrays(ee, sample, root / "cache", norm, config)
    masks = sensor_masks(arrays)
    observed = np.logical_or.reduce([v.any(axis=0)[0] for v in masks.values()])
    support = observed.reshape(40, 3, 40, 3).all(axis=(1, 3))
    rc = points[["output_row", "output_col"]].to_numpy()
    nvalid = int(support[rc[:, 0], rc[:, 1]].sum())
    if not eligible or not nvalid:
        atomic_json(
            rejection,
            dict(
                sample_id=sid,
                reason="input_or_point_coverage",
                total_points=len(points),
                input_valid_points=nvalid,
            ),
        )
        return sid
    base = root / "zarr" / sid
    # A shard without its record was interrupted before the pair commit. Only
    # this uncommitted sample is rebuilt; published samples are never touched.
    if base.exists():
        shutil.rmtree(base)
    schema = CoverPointSchema(observation_capacity=len(points))
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
            point_time_precision="date",
            label_support="point_not_pixel_area",
        ),
    )
    writer.write(arrays, point_targets(points))
    writer.finalize()
    atomic_json(
        committed,
        dict(
            row,
            input_path=f"zarr/{sid}/inputs.zarr",
            label_path=f"zarr/{sid}/labels.zarr",
            shard_index=0,
            input_valid_points=nvalid,
        ),
    )
    return sid


def pilot_ids(plan, points):
    """One window per glacier plus one containing each class; deterministic."""
    selected = set()
    for _, group in points.groupby("site_name"):
        selected.add(group.sample_id.value_counts().index[0])
    for _, group in points.groupby("cover_class"):
        selected.add(group.sample_id.value_counts().index[0])
    return plan[plan.sample_id.isin(selected)]


def collect(root, *, project, concurrency=4, pilot=False, ee=None, norm=None):
    if concurrency < 1 or concurrency > 8:
        raise ValueError("Concurrency must be between 1 and 8")
    root = Path(root)
    config = json.loads((root / "collection.json").read_text())
    if config["producer_identity"] != producer_identity():
        raise ValueError("Pixel/label production code differs from frozen preparation")
    plan, points = pd.read_parquet(root / "plan.parquet"), pd.read_parquet(root / "points.parquet")
    if ee is None:
        from hydrostate.data.gee.client import initialize

        ee = initialize(project)
    if norm is None:
        from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.data.normalize import (
            load_computed_config,
        )

        norm = load_computed_config()
    with locked(root / ".collection.lock", blocking=False):
        if (root / "_PUBLISHED.json").exists():
            raise ValueError("Already published; collection is immutable")
        # Require a successful representative audit before expanding the collection.
        if not pilot:
            gate = root / "audit" / "pilot.json"
            if not gate.exists() or not json.loads(gate.read_text()).get("passed"):
                raise ValueError("Run collect --pilot and audit --pilot before full collection")
        for name in ("records", "rejected", "cache"):
            (root / name).mkdir(exist_ok=True)
        normalization = root / "normalization.json"
        digest = fingerprint(norm)
        if normalization.exists() and json.loads(normalization.read_text())["identity"] != digest:
            raise ValueError("Pretrained normalization changed")
        atomic_json(normalization, dict(identity=digest))
        selected = pilot_ids(plan, points) if pilot else plan
        groups = {sid: frame for sid, frame in points.groupby("sample_id", sort=False)}
        pending = [
            row
            for row in selected.to_dict("records")
            if not (root / "records" / f"{row['sample_id']}.json").exists()
            and not (root / "rejected" / f"{row['sample_id']}.json").exists()
        ]

        def run(row):
            return collect_one(ee, row, groups[row["sample_id"]], root, config, norm)

        try:
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                for sid in pool.map(run, pending):
                    print(json.dumps(dict(sample_id=sid, status="processed")), flush=True)
        finally:
            records = [json.loads(p.read_text()) for p in sorted((root / "records").glob("*.json"))]
            if records:
                atomic_parquet(pd.DataFrame(records), root / "samples.parquet")
            rejected = len(list((root / "rejected").glob("*.json")))
            atomic_json(
                root / "collection_report.json",
                dict(
                    accepted=len(records),
                    rejected=rejected,
                    planned=len(plan),
                    complete=len(records) + rejected == len(plan),
                    pilot=pilot,
                ),
            )
    return json.loads((root / "collection_report.json").read_text())
