"""Rebalance ET by promoting whole evaluation groups; never expose training sites."""

import numpy as np
import pandas as pd


def rebalance_et(table, *, seed=43):
    """Preserve SM groups and train identities; target 15% ET per evaluation stratum.

    The remaining groups become train. Physical sites, not temporal records, are
    indivisible. A subset dynamic program optimizes window totals, then site counts;
    cover/climate/year/month balance resolves remaining ties across seeded starts.
    """
    if table.groupby("spatial_group_id").split.nunique().gt(1).any():
        raise ValueError("Existing spatial group crosses splits")
    if table.groupby("physical_group_id").spatial_group_id.nunique().gt(1).any():
        raise ValueError("Physical station crosses spatial groups")
    assignment = table.groupby("spatial_group_id").split.first().to_dict()
    sm_groups = set(table.loc[table.valid_sm.gt(0), "spatial_group_id"])
    et = table[table.valid_et.gt(0)]
    mixed = et.groupby("spatial_group_id").frequency_bin.nunique()
    # A colocated group spanning frequency bins stays intact in its original split.
    fixed_groups = sm_groups | set(mixed[mixed.gt(1)].index)
    rng = np.random.default_rng(seed)
    for frequency, full in et.groupby("frequency_bin"):
        target_windows = 0.15 * len(full)
        target_sites = 0.15 * full.physical_group_id.nunique()
        for split in ("validation", "test"):
            available = full[full.split.eq(split)]
            if available.empty:
                continue
            grouped = {k: v for k, v in available.groupby("spatial_group_id")}
            forced = set(grouped) & fixed_groups
            free = sorted(set(grouped) - forced)
            forced_windows = sum(len(grouped[k]) for k in forced)
            forced_sites = sum(grouped[k].physical_group_id.nunique() for k in forced)
            best = None
            for _ in range(4):
                order = list(rng.permutation(free))
                states = {(forced_windows, forced_sites): frozenset(forced)}
                for key in order:
                    n, ns = len(grouped[key]), grouped[key].physical_group_id.nunique()
                    new = {}
                    for (w, sites), selected in states.items():
                        location = (w + n, sites + ns)
                        if location not in states and location not in new:
                            new[location] = selected | {key}
                    states.update(new)
                primary = min(
                    (abs(w - target_windows), abs(ns - target_sites))
                    for (w, ns), keys in states.items()
                    if keys
                )
                for (w, ns), keys in states.items():
                    if not keys or (abs(w - target_windows), abs(ns - target_sites)) != primary:
                        continue
                    selected = available[available.spatial_group_id.isin(keys)]
                    diversity = 0.0
                    for col in ("landcover", "climate_class", "year", "month"):
                        expected = full[col].value_counts(normalize=True)
                        actual = (
                            selected[col]
                            .value_counts(normalize=True)
                            .reindex(expected.index, fill_value=0)
                        )
                        diversity += float(((expected - actual) ** 2).sum())
                    score = (*primary, diversity)
                    if best is None or score < best[0]:
                        best = score, keys
            retained = best[1]
            for key in grouped:
                assignment[key] = split if key in retained else "train"
    result = table.copy()
    result["previous_split"] = result.split
    result["split"] = result.spatial_group_id.map(assignment)
    if result.loc[result.previous_split.eq("train"), "split"].ne("train").any():
        raise AssertionError("Training group leaked into evaluation")
    if (
        result.loc[result.valid_sm.gt(0), "split"]
        .ne(result.loc[result.valid_sm.gt(0), "previous_split"])
        .any()
    ):
        raise AssertionError("SM split changed")
    return result


def collect_additive_targets(
    selected, rows, failures, config, collect, consume, *, workers=4, max_attempts=None
):
    """Bounded waves; same-split/frequency replacements until success targets or exhaustion."""
    from collections import Counter
    from concurrent.futures import ThreadPoolExecutor

    from .candidates import StratifiedPicker
    from .production import table_records
    from .temporal import epoch

    if config["min_date_gap_days"] != 0:
        raise ValueError("Additive ET plan explicitly uses zero date gap")
    targets = config["success_targets_by_split"]
    if set(targets) != {"train", "validation", "test"} or any(n < 0 for n in targets.values()):
        raise ValueError("Expected nonnegative three-way success targets")
    reference = pd.read_parquet(config["sampling_reference"])
    reference = reference[reference.valid_et.gt(0)]
    lookup = {r["candidate_id"]: r for r in table_records(selected)}
    pickers, quotas = {}, {}
    for split, total in targets.items():
        candidates = [r for r in lookup.values() if r["split"] == split]
        picker = StratifiedPicker(candidates, seed=config["seed"], gap_days=0)
        for r in table_records(reference[reference.split.eq(split)]):
            picker.timestamps[r["candidate_id"]] = epoch(r["target_time"])
            picker.record(r, success=True)
        for r in rows + failures:
            if r["split"] == split:
                picker.record(lookup[r["candidate_id"]], success=r["status"] == "success")
        pickers[split] = picker
        quotas[split] = {b: len(reference) + total + 1 for b in picker.buckets}
    counts = Counter(r["split"] for r in rows)
    processed = 0
    exhausted = set()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        while any(counts[s] < n for s, n in targets.items()):
            wave, pending = [], Counter()
            while len(wave) < workers and (
                max_attempts is None or processed + len(wave) < max_attempts
            ):
                available = [
                    s
                    for s, n in targets.items()
                    if s not in exhausted and counts[s] + pending[s] < n
                ]
                if not available:
                    break
                split = min(available, key=lambda s: (counts[s] + pending[s]) / max(targets[s], 1))
                row = pickers[split].next(quotas[split])
                if row is None:
                    exhausted.add(split)
                    continue
                pickers[split].record(row, success=False)
                pending[split] += 1
                wave.append(row)
            if not wave:
                break
            for row, result in zip(wave, pool.map(collect, wave), strict=True):
                success = result["status"] == "success"
                pickers[row["split"]].record(row, success=success)
                counts[row["split"]] += int(success)
                consume(result)
                processed += 1
    return all(counts[s] >= n for s, n in targets.items())


def prepare_revision(published, audit, parent, output, split_source, *, additional=1200):
    """Write metadata-only revision and an additive, deduplicated ET candidate pool."""
    import json
    import shutil
    from pathlib import Path

    from hydrostate.data.gee.contracts import atomic_json

    from .archives import sha256

    published, parent, output, split_source = map(Path, (published, parent, output, split_source))
    if output.exists():
        raise FileExistsError(output)
    original = pd.read_parquet(published / "inventory.parquet")
    stations = json.loads(Path(audit).read_text())
    passing = {r["station"] for r in stations if r["results"]["measured_all"]["status"] == "pass"}
    if ((original.valid_sm > 0) & (original.valid_et > 0)).any():
        raise ValueError("Mixed-target rows need per-target EBR masking")
    retained = original[original.valid_et.eq(0) | original.physical_group_id.isin(passing)]
    revised = rebalance_et(retained)
    assignment = revised.groupby("spatial_group_id").split.first()
    station_map = pd.read_parquet(published / "station_splits.parquet")
    station_map["split"] = station_map.spatial_group_id.map(assignment).fillna(station_map.split)
    if station_map.loc[station_map.product_overlap_count.gt(0), "split"].ne("train").any():
        raise ValueError("Product overlap in evaluation")
    candidates = pd.read_parquet(parent / "input_candidates.parquet")
    candidates = candidates[
        candidates.physical_group_id.isin(passing)
        & candidates.has_et
        & ~candidates.has_sm
        & candidates.target_time.ge("2015-01-01")
        & candidates.target_end.le("2026-01-01")
    ].copy()
    keys = ["physical_group_id", "target_time", "target_end"]
    old_keys = pd.MultiIndex.from_frame(original[keys])
    candidates = candidates[~pd.MultiIndex.from_frame(candidates[keys]).isin(old_keys)]
    candidates = candidates.sort_values(
        ["has_et_footprint", "source_priority", "candidate_id"], ascending=[False, True, True]
    ).drop_duplicates(keys)
    candidates["split"] = candidates.spatial_group_id.map(assignment)
    if candidates.split.isna().any():
        raise ValueError("New spatial groups are outside this revision")
    candidates["draw_order"] = range(len(candidates))
    members = pd.read_parquet(parent / "label_members.parquet")
    members = members[members.input_candidate_id.isin(candidates.candidate_id)].copy()
    members["split"] = members.input_candidate_id.map(candidates.set_index("candidate_id").split)
    et = revised[revised.valid_et.gt(0)]
    total = len(et) + additional
    desired = dict(validation=round(total * 0.15), test=round(total * 0.15))
    desired["train"] = total - sum(desired.values())
    targets = {s: int(n - et.split.eq(s).sum()) for s, n in desired.items()}
    if min(targets.values()) < 0:
        raise ValueError("Requested increment cannot achieve target proportions")
    output.mkdir(parents=True)
    revised.to_parquet(output / "inventory.parquet", index=False)
    datasets = []
    source_datasets = json.loads((published / "datasets.json").read_text())["datasets"]
    for cohort, table in revised.groupby("cohort"):
        name = f"cohort_{cohort}.parquet"
        table.to_parquet(output / name, index=False)
        root = published / source_datasets[int(cohort)]["data_root"]
        datasets.append(dict(type="SiteDataset", manifest=str(output / name), data_root=str(root)))
    atomic_json(
        output / "datasets.json", dict(datasets=datasets, statistics_status="pending_recompute")
    )
    split_root = output / "splits"
    split_root.mkdir()
    for name in (
        "split_config.json",
        "product_support.json",
        "product_overlaps.parquet",
        "joint_product_selection.parquet",
    ):
        shutil.copyfile(split_source / name, split_root / name)
    station_map.to_parquet(split_root / "station_splits.parquet", index=False)
    plan = output / "plan"
    plan.mkdir()
    for name in ("input_candidates", "selected"):
        candidates.to_parquet(plan / f"{name}.parquet", index=False)
    members.to_parquet(plan / "label_members.parquet", index=False)
    report = dict(
        baseline_et=len(et),
        baseline_by_split=et.split.value_counts().to_dict(),
        additional_targets=targets,
        final_targets=desired,
        pool_by_split=candidates.split.value_counts().to_dict(),
        pool_by_frequency=candidates.frequency_bin.value_counts().to_dict(),
        unchanged_sm=True,
        minimum_date_gap_days=0,
        audit_sha256=sha256(audit),
        parent_inventory_sha256=sha256(published / "inventory.parquet"),
        status="prepared_not_published",
        product_exclusions="conservatively_retained",
        split_rule="whole_groups_eval_to_train_only_sm_fixed",
    )
    atomic_json(output / "revision.json", report)
    atomic_json(split_root / "revision.json", report)
    return report
