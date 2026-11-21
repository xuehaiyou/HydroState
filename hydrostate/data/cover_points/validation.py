"""Full point/grid/array audit, with a separate gate for representative real acquisition."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

from hydrostate.data.gee.contracts import atomic_json
from hydrostate.data.retrieval.alignment import sensor_masks
from hydrostate.storage.packs import file_hash

from .collection import pilot_ids, point_targets
from .contract import CLASS_MAP, VERSION


def audit(root, *, pilot=False, publish=False):
    from hydrostate.datasets.cover_point_dataset import CoverPointDataset

    root = Path(root)
    data = CoverPointDataset(root, require_published=False)
    points, plan = pd.read_parquet(root / "points.parquet"), pd.read_parquet(root / "plan.parquet")
    if not set(data.table.sample_id) <= set(plan.sample_id):
        raise ValueError("Unplanned samples")
    groups = {sid: group for sid, group in points.groupby("sample_id")}
    seen, supported = [], []
    for i in range(len(data)):
        item = data[i]
        labels = item["points"]
        expected = point_targets(groups[item["sample_id"]])
        if any(not np.array_equal(labels[k], v) for k, v in expected.items()):
            raise ValueError("Point source/array disagreement")
        rc = labels["cover_point_rc"]
        if np.any((rc < 0) | (rc >= 40)):
            raise ValueError("Point index outside output grid")
        row = data.table.iloc[i]
        tr = json.loads(row["transform"])
        x, y = Transformer.from_crs(4326, row.crs, always_xy=True).transform(
            *labels["cover_point_lonlat"].T
        )
        recomputed = np.stack([np.floor((tr[5] - y) / 30), np.floor((x - tr[2]) / 30)], axis=1)
        if not np.array_equal(recomputed, rc):
            raise ValueError("Coordinate to output grid mismatch")
        if not np.array_equal(
            labels["cover_point_class"],
            [CLASS_MAP[int(c)] for c in labels["cover_point_original_class"]],
        ):
            raise ValueError("Class mapping mismatch")
        start, end = item["inputs"]["target_bounds"].numpy()
        dates = labels["cover_point_date"]
        if np.any((dates < start) | (dates >= end)):
            raise ValueError("Point date outside target support")
        masks = sensor_masks(
            {k: v.numpy() for k, v in item["inputs"].items() if k != "target_time"}
        )
        observed = np.logical_or.reduce([v.any(axis=0)[0] for v in masks.values()])
        support = observed.reshape(40, 3, 40, 3).all(axis=(1, 3))
        supported.extend(labels["cover_point_index"][support[rc[:, 0], rc[:, 1]]].tolist())
        for sensor in ("s1", "s2", "landsat"):
            values = item["inputs"][sensor].numpy()
            if not np.isfinite(values).all():
                raise ValueError("Nonfinite normalized inputs")
        seen.extend(labels["cover_point_index"].tolist())
    if len(seen) != len(set(seen)):
        raise ValueError("Point scored in multiple windows")
    rejected = {p.stem for p in (root / "rejected").glob("*.json")}
    accepted = set(data.table.sample_id)
    if accepted & rejected or not rejected <= set(plan.sample_id):
        raise ValueError("Inconsistent rejection ledger")
    required = set(pilot_ids(plan, points).sample_id) if pilot else set(plan.sample_id)
    complete = required <= accepted | rejected
    usable = points[points.point_index.isin(supported)]
    representative = set(usable.site_name) == set(points.site_name) and set(
        usable.cover_class
    ) == set(points.cover_class)
    if not complete or (pilot and not representative):
        raise ValueError("Incomplete audit scope or missing pilot sites/classes")
    cell = ["sample_id", "observation_date", "output_row", "output_col"]
    report = dict(
        passed=True,
        version=VERSION,
        scope="pilot" if pilot else "full",
        accepted_windows=len(data),
        rejected_windows=len(rejected),
        label_points=len(seen),
        input_supported_points=len(supported),
        uncollected_points=len(points) - len(seen),
        conflicting_cells=int(points.groupby(cell).cover_class.nunique().gt(1).sum()),
        supported_class_counts={
            str(k): int(v) for k, v in usable.cover_class.value_counts().items()
        },
        sites=sorted(usable.site_name.unique()),
        training_overlap="must_be_audited_per_checkpoint_before_independent_scoring",
    )
    atomic_json(root / "audit" / ("pilot.json" if pilot else "full.json"), report)
    if publish:
        if pilot:
            raise ValueError("Cannot publish pilot scope")
        atomic_json(
            root / "_PUBLISHED.json",
            dict(
                version=VERSION,
                complete=True,
                samples_sha256=file_hash(root / "samples.parquet"),
                points_sha256=file_hash(root / "points.parquet"),
                collection_sha256=file_hash(root / "collection.json"),
                report=report,
            ),
        )
    return report
