"""Publication audits bind labels, real inputs, region splits and training exclusions."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from hydrostate.data.cover_points.prepare import atomic_parquet
from hydrostate.data.gee.contracts import atomic_json
from hydrostate.data.retrieval.alignment import sensor_masks
from hydrostate.data.site.splits import smap_circle, tile_circle
from hydrostate.storage.packs import file_hash

from .contract import CLASSES, VERSION
from .pipeline import configuration, selected_pilot
from .splits import exclusions


def audit_training(root, manifests):
    root, _, plan = configuration(root)
    rows, bindings = [], {}
    for path in manifests:
        path = Path(path).resolve()
        table = pd.read_parquet(path)
        if "split" not in table:
            raise ValueError("Reference manifest has no split")
        table = table[table.split.eq("train")]
        if table.empty:
            raise ValueError(f"No training records: {path}")
        bindings[str(path)] = file_hash(path)
        for r in table.to_dict("records"):
            common = dict(sample_id=r["sample_id"], manifest=str(path))
            rows.append(dict(tile_circle(r), **common))
            counts = r.get("valid_label_counts")
            if isinstance(counts, dict) and counts.get("soil_moisture_valid", 0):
                attrs = json.loads((path.parent / r["label_path"] / ".zattrs").read_text())
                rows.append(dict(smap_circle(attrs["sm_quality"]["support"]), **common))
            # Site input circles + 35 km margin conservatively exceed local SM/ET footprints.
    reference = pd.DataFrame(rows).drop_duplicates(["manifest", "sample_id", "kind"])
    excluded, distances = exclusions(plan, reference)
    atomic_parquet(reference, root / "audit/training_support.parquet")
    atomic_parquet(excluded, root / "training_exclusions.parquet")
    atomic_parquet(distances, root / "audit/training_distances.parquet")
    report = dict(
        passed=True,
        reference_manifests=bindings,
        support_records=len(reference),
        excluded_samples=len(excluded.drop_duplicates(["manifest", "sample_id"])),
        policy="apply_exclusions_to_future_training_not_retroactive_checkpoint_independence",
    )
    atomic_json(root / "audit/training.json", report)
    return report


def audit(root, pilot=False, publish=False):
    from hydrostate.datasets.manual_cover_dataset import ManualCoverDataset

    root, config, plan = configuration(root)
    data = ManualCoverDataset(root, require_published=False)
    inventory = pd.read_parquet(root / "label_inventory.parquet")
    if data.table.empty:
        raise ValueError("No inputs to audit")
    required = (
        set(selected_pilot(inventory[inventory.label_usable])) if pilot else set(plan.sample_id)
    )
    rejected = {p.stem for p in (root / "rejected").glob("*.json")}
    accepted = set(data.table.sample_id)
    if accepted & rejected or not (accepted | rejected) <= set(plan.sample_id):
        raise ValueError("Invalid collection ledger")
    if not required <= accepted | rejected:
        raise ValueError("Collection scope not finished")
    if pilot and not required <= accepted:
        raise ValueError(
            "Representative pilot input failed; inspect and choose a new pilot explicitly"
        )
    merged = data.table.merge(
        plan[["sample_id", "split", "spatial_group"]],
        on="sample_id",
        suffixes=("", "_planned"),
        validate="one_to_one",
    )
    if not (
        merged.split.eq(merged.split_planned)
        & merged.spatial_group.eq(merged.spatial_group_planned)
    ).all():
        raise ValueError("Samples changed split")
    stats = []
    for i in range(len(data)):
        row = data.table.iloc[i]
        item = data[i]
        labels = item["data_samples"]
        counts = {}
        for name in CLASSES:
            v, valid = labels[name].numpy(), labels[name + "_valid"].numpy()
            if not np.isfinite(v).all() or ((v < -1e-6) | (v > 1 + 1e-6)).any():
                raise ValueError("Invalid proportion")
            counts[name + "_pixels"] = int(valid.sum())
            counts[name + "_equivalent_pixels"] = float(v[valid].sum())
        if row.source == "sen1floods11":
            if counts["snow_ice_pixels"] or counts["land_pixels"]:
                raise ValueError("Nonwater incorrectly assigned land/ice truth")
            area = labels["valid_area_fraction"].numpy()
            if np.any(
                labels["water_valid"].numpy() & (area < 1 - config["flood_full_valid_tolerance"])
            ):
                raise ValueError("Partial annotation used as full-pixel truth")
        else:
            masks = [labels[c + "_valid"].numpy() for c in CLASSES]
            if not all(np.array_equal(masks[0], m) for m in masks):
                raise ValueError("Glacier one-hot masks disagree")
            total = sum(labels[c].numpy() for c in CLASSES)
            if not np.allclose(total[masks[0]], 1):
                raise ValueError("Glacier proportions do not close")
            if np.any(masks[0] & labels["point_conflict"].numpy().astype(bool)):
                raise ValueError("Conflicting cell used for supervision")
        inputs = {k: v.numpy() for k, v in item["inputs"].items() if k != "target_time"}
        if not all(np.isfinite(v).all() for v in inputs.values()):
            raise ValueError("Nonfinite input")
        observed = np.logical_or.reduce([v.any(axis=0)[0] for v in sensor_masks(inputs).values()])
        support = observed.reshape(40, 3, 40, 3).all(axis=(1, 3))
        counts["input_supported_water_label_pixels"] = int(
            (support & labels["water_valid"].numpy()[0]).sum()
        )
        stats.append(
            dict(
                sample_id=row.sample_id,
                source=row.source,
                split=row.split,
                region=row.region,
                **counts,
            )
        )
    stats = pd.DataFrame(stats)
    if stats.input_supported_water_label_pixels.eq(0).any():
        raise ValueError("Accepted sample has no input-supported label")
    atomic_parquet(
        stats, root / "audit" / ("pilot_samples.parquet" if pilot else "samples.parquet")
    )
    report = dict(
        passed=True,
        pilot=pilot,
        accepted=len(accepted),
        rejected=len(rejected),
        summary=stats.groupby(["source", "split"])
        .sum(numeric_only=True)
        .reset_index()
        .to_dict("records"),
    )
    atomic_json(root / "audit" / ("pilot.json" if pilot else "full.json"), report)
    if publish:
        if pilot:
            raise ValueError("Cannot publish pilot")
        training = json.loads((root / "audit/training.json").read_text())
        if not training["passed"]:
            raise ValueError("Training support audit failed")
        for manifest, digest in training["reference_manifests"].items():
            if file_hash(manifest) != digest:
                raise ValueError("Audited training manifest changed")
        if set(data.table.split) != {"train", "val", "test"}:
            raise ValueError("Missing split after input filtering")
        for source in ("glacier", "sen1floods11"):
            if set(data.table[data.table.source.eq(source)].split) != {"train", "val", "test"}:
                raise ValueError("Source missing a split")
        atomic_parquet(data.table, root / "inventory.parquet")
        for (source, split), table in data.table.groupby(["source", "split"]):
            atomic_parquet(table, root / f"cohort_{source}_{split}.parquet")
        names = [
            "samples.parquet",
            "inventory.parquet",
            "plan.parquet",
            "splits.parquet",
            "collection.json",
            "points.parquet",
            "training_exclusions.parquet",
            "audit/training.json",
            "sources/flood_catalog.json",
        ]
        atomic_json(
            root / "_PUBLISHED.json",
            dict(
                version=VERSION,
                complete=True,
                metadata_hashes={n: file_hash(root / n) for n in names},
                report=report,
            ),
        )
    return report
