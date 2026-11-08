"""Versioned frequency-stratified spatial-group splits over completed site cohorts."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from hydrostate.data.gee.contracts import atomic_json

from .archives import sha256

SPLITS = ("train", "validation", "test")


def task_coverage(table, locked_train=()):
    """Count independent supports, not sensors or dates; expose infeasible strata."""
    locked = set(locked_train)
    result = []
    for frequency in range(7):
        for task in ("sm", "et"):
            subset = table[table.frequency_bin.eq(frequency) & table[f"valid_{task}"].gt(0)]
            groups = set(subset.spatial_group_id)
            free = groups - locked
            feasible = len(groups) >= 3 and len(free) >= 2
            # Preserve at least one training group; prefer two evaluation groups when possible.
            evaluation = min(2, (len(groups) - 1) // 2, len(free) // 2) if feasible else 0
            result.append(
                dict(
                    frequency_bin=frequency,
                    task=task,
                    groups=sorted(groups),
                    spatial_groups=len(groups),
                    evaluation_eligible_groups=len(free),
                    three_way_feasible_by_count=feasible,
                    minimum_groups=dict(
                        train=1 if groups else 0, validation=evaluation, test=evaluation
                    ),
                )
            )
    return result


def stratify_groups(table, locked_train=(), *, seed=43, ratios=(0.7, 0.15, 0.15)):
    """Require task coverage before optimizing frequency proportions and diversity.

    Whole spatial groups are indivisible. Ratios are soft targets, never permission
    to move overlapping product supports into evaluation or split a physical site.
    """
    ratios = np.asarray(ratios, dtype=float)
    if ratios.shape != (3,) or np.any(ratios <= 0) or not np.isclose(ratios.sum(), 1):
        raise ValueError("Expected three positive split fractions summing to one")
    if table.groupby("physical_group_id").spatial_group_id.nunique().gt(1).any():
        raise ValueError("Physical station crosses spatial groups")
    keys = sorted(table.spatial_group_id.unique())
    coverage = task_coverage(table, locked_train)
    incidence = np.array([[int(key in item["groups"]) for item in coverage] for key in keys])
    minimum = np.array([[item["minimum_groups"][s] for item in coverage] for s in SPLITS])
    primary = pd.crosstab(table.spatial_group_id, table.frequency_bin).reindex(keys).values
    secondary = []
    for target in ("valid_sm", "valid_et"):
        subset = table[table[target].gt(0)]
        secondary.append(
            pd.crosstab(subset.spatial_group_id, subset.frequency_bin)
            .reindex(keys, fill_value=0)
            .values
        )
    for column in ("landcover", "climate_class", "year", "month"):
        secondary.append(
            pd.crosstab(table.spatial_group_id, table[column]).reindex(keys, fill_value=0).values
        )
    secondary = np.concatenate(secondary, axis=1)
    p = primary / primary.sum(axis=0).clip(1)
    s = secondary / secondary.sum(axis=0).clip(1)
    features = np.concatenate([p, s, incidence], axis=1)
    width = p.shape[1]
    target = ratios[:, None]
    fixed = np.array([key in set(locked_train) for key in keys])

    def score(counts):
        errors = np.square(counts[:, : -len(coverage)] - target)
        missing = np.maximum(minimum - counts[:, -len(coverage) :], 0)
        return (
            int(missing.sum()),
            round(float(errors[:, :width].sum()), 12),
            float(errors[:, width:].mean()),
        )

    rng = np.random.default_rng(seed)
    best = None
    for _ in range(8):
        assignments = np.zeros(len(keys), dtype=int)
        counts = np.zeros((3, features.shape[1]))
        counts[0] = features[fixed].sum(axis=0)
        # Rare frequency groups and large indivisible stations are placed first.
        priority = p.max(axis=1) + rng.uniform(0, 1e-9, len(keys))
        order = [i for i in np.argsort(-priority) if not fixed[i]]
        for i in order:
            choices = []
            for split in range(3):
                trial = counts.copy()
                trial[split] += features[i]
                choices.append(score(trial))
            assignments[i] = min(range(3), key=lambda j: choices[j])
            counts[assignments[i]] += features[i]
        for _ in range(12):
            changed = False
            for i in rng.permutation(order):
                old = assignments[i]
                optimum, dest = score(counts), old
                for split in range(3):
                    trial = counts.copy()
                    trial[old] -= features[i]
                    trial[split] += features[i]
                    value = score(trial)
                    if value < optimum:
                        optimum, dest = value, split
                if dest != old:
                    counts[old] -= features[i]
                    counts[dest] += features[i]
                    assignments[i], changed = dest, True
            if not changed:
                break
        value = score(counts)
        if best is None or value < best[0]:
            best = value, assignments.copy()
    if best[0][0]:
        raise ValueError(
            "No jointly feasible task-covering assignment found; inspect spatial groups"
        )
    return dict(zip(keys, (SPLITS[i] for i in best[1])))


def resplit_cohorts(roots, split_root, output):
    """Write selection manifests; original Zarr and production provenance stay immutable."""
    roots = list(map(Path, roots))
    split_root, output = Path(split_root), Path(output)
    if output.exists():
        raise FileExistsError(output)
    frames = []
    for i, root in enumerate(roots):
        report = json.loads((root / "collection_report.json").read_text())
        if not report["initial_screening_complete"]:
            raise ValueError("Wait for every cohort to finish screening")
        table = pd.read_parquet(root / "samples.parquet")
        if len(table) != report["successful"]:
            raise ValueError("Incomplete manifest")
        frames.append(table.assign(cohort=i, original_split=table.split))
    table = pd.concat(frames, ignore_index=True)
    if table.sample_id.duplicated().any() or table.candidate_id.duplicated().any():
        raise ValueError("Cohorts contain duplicate samples")
    stations = pd.read_parquet(split_root / "station_splits.parquet")
    known = stations.dropna(subset=["spatial_group_id"])
    mapping = known.groupby("physical_group_id").spatial_group_id.agg(set).to_dict()
    for row in table[["physical_group_id", "spatial_group_id"]].drop_duplicates().itertuples():
        if row.spatial_group_id not in mapping.get(row.physical_group_id, set()):
            raise ValueError("Unknown spatial group lineage")
    mandatory = known.product_overlap_count.fillna(0).gt(0) | known.split_reason.eq(
        "previous_development_group"
    )
    locked = set(known.loc[mandatory, "spatial_group_id"])
    assignment = stratify_groups(table, locked)
    table["split"] = table.spatial_group_id.map(assignment)
    coverage = task_coverage(table, locked)
    for item in coverage:
        subset = table[
            table.frequency_bin.eq(item["frequency_bin"]) & table[f"valid_{item['task']}"].gt(0)
        ]
        item["actual_groups"] = {
            s: subset[subset.split.eq(s)].spatial_group_id.nunique() for s in SPLITS
        }
        if any(item["actual_groups"][s] < item["minimum_groups"][s] for s in SPLITS):
            raise AssertionError("Task coverage requirement violated")
    if table.loc[table.spatial_group_id.isin(locked), "split"].ne("train").any():
        raise AssertionError("Product/development overlap leaked into evaluation")
    output.mkdir(parents=True)
    datasets = []
    for i, root in enumerate(roots):
        selected = table[table.cohort.eq(i)].copy()
        manifest = output / f"cohort_{i}.parquet"
        selected.to_parquet(manifest, index=False)
        datasets.append(
            dict(
                type="SiteDataset", manifest=str(manifest.resolve()), data_root=str(root.resolve())
            )
        )
    # Combined inventory is for reporting; loading retains each cohort's config hash.
    table.to_parquet(output / "inventory.parquet", index=False)
    atomic_json(output / "datasets.json", dict(datasets=datasets))
    station_map = known.copy()
    station_map["original_split"] = station_map.split
    station_map["split"] = station_map.spatial_group_id.map(assignment).fillna(station_map.split)
    station_map.to_parquet(output / "station_splits.parquet", index=False)
    rows = []
    for (frequency, split), group in table.groupby(["frequency_bin", "split"]):
        rows.append(
            dict(
                frequency_bin=int(frequency),
                split=split,
                windows=len(group),
                physical_sites=group.physical_group_id.nunique(),
                spatial_groups=group.spatial_group_id.nunique(),
                sm_windows=int(group.valid_sm.gt(0).sum()),
                et_windows=int(group.valid_et.gt(0).sum()),
            )
        )
    report = dict(
        version="site_frequency_task_spatial_split_v3",
        task_coverage=coverage,
        seed=43,
        target_ratios=dict(zip(SPLITS, [0.7, 0.15, 0.15])),
        distribution=rows,
        samples=len(table),
        changed_windows=int(table.split.ne(table.original_split).sum()),
        source_manifests={str(r): sha256(r / "samples.parquet") for r in roots},
        source_splits_sha256=sha256(split_root / "station_splits.parquet"),
        joint_product_selection=str(split_root / "joint_product_selection.parquet"),
        joint_product_selection_sha256=sha256(split_root / "joint_product_selection.parquet"),
        evaluation_use="New split version; historical diagnostics are not unseen evaluation",
        statistics_status="pending_train_only_recompute",
        published=False,
    )
    atomic_json(output / "split_report.json", report)
    return report
