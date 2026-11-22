"""Append formerly date-excluded flood labels to an unpublished manual release.

Run this file with PYTHONPATH pointing to the original frozen producer. Only
the source selection changes; existing pixel production and hashes stay intact.
Stop collection before use. Existing IDs and splits are never regenerated.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from hydrostate.data.cover_points.prepare import atomic_parquet, parse_points
from hydrostate.data.gee.contracts import atomic_json, locked
from hydrostate.data.manual_cover.pipeline import configuration
from hydrostate.data.manual_cover.prepare import flood_plan
from hydrostate.data.site.splits import GEOD, tile_circle
from hydrostate.storage.packs import file_hash


def append_train(plan, added, margin=35000):
    """Preserve old partitions; refuse a new train region near any holdout."""
    added = added.copy()
    added["split"] = "train"
    added["spatial_group"] = added.region
    combined = pd.concat([plan, added], ignore_index=True)
    circles = {}
    for region, frame in combined.groupby("region"):
        lon, lat = float(frame.lon.mean()), float(frame.lat.mean())
        distances = GEOD.inv(
            np.full(len(frame), lon),
            np.full(len(frame), lat),
            frame.lon.to_numpy(),
            frame.lat.to_numpy(),
        )[2]
        radius = max(distances) + max(tile_circle(r)["radius_m"] for r in frame.to_dict("records"))
        if frame.split.nunique() != 1:
            raise ValueError("Region crosses splits")
        circles[region] = (lon, lat, radius, frame.split.iloc[0])
    report = []
    names = sorted(circles)
    for i, a in enumerate(names):
        for b in names[:i]:
            ca, cb = circles[a], circles[b]
            gap = GEOD.inv(ca[0], ca[1], cb[0], cb[1])[2] - ca[2] - cb[2]
            different = ca[3] != cb[3]
            if different and gap <= margin:
                raise ValueError(f"New allocation violates spatial isolation: {a}, {b}")
            report.append(
                dict(region_a=a, region_b=b, support_gap_m=gap, different_split=different)
            )
    return combined, pd.DataFrame(report)


def extend(root):
    root, config, plan = configuration(root)
    backup = root / "amendments/remove_year_cutoff"
    if backup.exists():
        raise FileExistsError("Amendment already started; inspect its backup before retrying")
    catalog = json.loads((root / "sources/flood_catalog.json").read_text())
    additions = [r for r in catalog if r["exclusion_reason"] == "outside_2017_2024_input_contract"]
    if not additions:
        raise ValueError("No date-excluded sources to append")
    for row in additions:
        row.update(included=True, exclusion_reason="")
        if (root / row["path"]).stat().st_size != row["size_bytes"]:
            raise ValueError("Source size changed")
    added = flood_plan(additions)
    if set(added.sample_id) & set(plan.sample_id):
        raise ValueError("Addition duplicates an existing sample")
    combined, distances = append_train(plan, added, config["split_margin_m"])
    glacier = Path(config["glacier_root"])
    raw = glacier / "sources/Sentinel-2_TOA_training_data.csv"
    parsed, duplicates = parse_points(raw)
    existing = pd.read_parquet(root / "points.parquet")
    columns = ["source_row", "site_name", "observation_date", "lon", "lat", "original_class"]
    pd.testing.assert_frame_equal(
        parsed[columns].sort_values("source_row").reset_index(drop=True),
        existing[columns].sort_values("source_row").reset_index(drop=True),
        check_dtype=False,
    )
    files = [
        "plan.parquet",
        "splits.parquet",
        "sources/flood_catalog.json",
        "audit/region_distances.parquet",
        "audit/preparation.json",
        "_LABELS_READY.json",
        "label_inventory.parquet",
        "audit/pilot.json",
        "audit/training.json",
        "collection.json",
    ]
    with (
        locked(root / ".collection.lock", blocking=False),
        locked(root / ".labels.lock", blocking=False),
    ):
        backup.mkdir(parents=True)
        hashes = {}
        for name in files:
            path = root / name
            if path.exists():
                dest = backup / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, dest)
                hashes[name] = file_hash(path)
        report = dict(
            complete=False,
            original_hashes=hashes,
            added_chips=len(additions),
            added_windows=len(added),
            original_windows=len(plan),
            total_windows=len(combined),
            unchanged_version=config["version"],
            unchanged_producer=config["producer_identity"],
            extension_code_sha256=file_hash(Path(__file__)),
            glacier_points=len(parsed),
            glacier_duplicates=duplicates,
            glacier_date_range=[parsed.observation_date.min(), parsed.observation_date.max()],
            glacier_missing_points=0,
            policy="no_fixed_year_cutoff; actual_input_availability",
        )
        atomic_json(backup / "report.json", report)
        for name in ["_LABELS_READY.json", "audit/pilot.json", "audit/training.json"]:
            (root / name).unlink(missing_ok=True)
        atomic_json(root / "sources/flood_catalog.json", catalog)
        atomic_parquet(combined, root / "plan.parquet")
        atomic_parquet(
            combined[["sample_id", "source", "region", "spatial_group", "split"]],
            root / "splits.parquet",
        )
        atomic_parquet(distances, root / "audit/region_distances.parquet")
        preparation = json.loads((root / "audit/preparation.json").read_text())
        preparation.update(
            windows=len(combined),
            included_flood_chips=sum(r["included"] for r in catalog),
            region_splits=combined.groupby(["source", "region", "split"])
            .size()
            .reset_index(name="planned_windows")
            .to_dict("records"),
        )
        atomic_json(root / "audit/preparation.json", preparation)
        report["complete"] = True
        report["new_plan_sha256"] = file_hash(root / "plan.parquet")
        atomic_json(backup / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    print(json.dumps(extend(parser.parse_args().root)), flush=True)
