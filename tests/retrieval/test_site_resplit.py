import json

import pandas as pd
import pytest

from hydrostate.data.site.resplit import resplit_cohorts, stratify_groups


def samples():
    rows = []
    for frequency in (0, 1):
        for i in range(20):
            for day in range(2):
                rows.append(
                    dict(
                        candidate_id=f"{frequency}_{i}_{day}",
                        sample_id=f"{frequency}_{i}_{day}",
                        physical_group_id=f"p{frequency}_{i}",
                        spatial_group_id=f"s{frequency}_{i}",
                        frequency_bin=frequency,
                        valid_sm=1,
                        valid_et=i % 2,
                        landcover=i % 3,
                        climate_class=i % 2,
                        year=2020,
                        month=1,
                        split="test",
                    )
                )
    return pd.DataFrame(rows)


def test_frequency_ratios_indivisible_groups_and_product_lock():
    table = samples()
    assignment = stratify_groups(table, {"s1_0", "s0_0"})
    assert assignment == stratify_groups(table, {"s1_0", "s0_0"})
    table["split"] = table.spatial_group_id.map(assignment)
    assert assignment["s1_0"] == assignment["s0_0"] == "train"
    assert table.groupby("physical_group_id").split.nunique().max() == 1
    for _, group in table.groupby("frequency_bin"):
        assert group.split.value_counts().to_dict() == dict(train=28, validation=6, test=6)


def test_completed_cohorts_are_selection_manifests_not_mutated_zarr(tmp_path):
    base, split, out = [tmp_path / name for name in ("base", "split", "out")]
    base.mkdir()
    split.mkdir()
    table = samples()
    table.to_parquet(base / "samples.parquet", index=False)
    (base / "collection_report.json").write_text(
        json.dumps(dict(initial_screening_complete=False, successful=len(table)))
    )
    with pytest.raises(ValueError, match="finish screening"):
        resplit_cohorts([base], split, out)
    assert not out.exists()
    (base / "collection_report.json").write_text(
        json.dumps(dict(initial_screening_complete=True, successful=len(table)))
    )
    stations = table[["physical_group_id", "spatial_group_id", "split"]].drop_duplicates()
    stations["product_overlap_count"] = 0
    stations["split_reason"] = "seeded_group_assignment"
    stations.loc[stations.spatial_group_id.eq("s1_0"), "product_overlap_count"] = 1
    stations.to_parquet(split / "station_splits.parquet", index=False)
    pd.DataFrame({"sample_id": ["product"]}).to_parquet(split / "joint_product_selection.parquet")
    report = resplit_cohorts([base], split, out)
    assert report["samples"] == len(table)
    assert pd.read_parquet(base / "samples.parquet").split.eq("test").all()
    revised = pd.read_parquet(out / "cohort_0.parquet")
    assert revised[revised.spatial_group_id.eq("s1_0")].split.eq("train").all()
    assert revised.original_split.eq("test").all()
    datasets = json.loads((out / "datasets.json").read_text())
    assert datasets["datasets"][0]["data_root"] == str(base)


def test_rare_tasks_have_independent_train_validation_and_test_supports():
    # Three SM groups and five ET groups, with the largest group supporting both.
    rows = []
    for i, (sm, et) in enumerate([(144, 58), (77, 0), (64, 0), (0, 6), (0, 3), (0, 7), (0, 8)]):
        for j in range(sm + et):
            rows.append(
                dict(
                    spatial_group_id=str(i),
                    physical_group_id=str(i),
                    frequency_bin=1,
                    valid_sm=int(j < sm),
                    valid_et=int(j >= sm),
                    landcover=1,
                    climate_class=1,
                    year=2020,
                    month=1,
                )
            )
    table = pd.DataFrame(rows)
    table["split"] = table.spatial_group_id.map(stratify_groups(table))
    for split in ("train", "validation", "test"):
        group = table[table.split.eq(split)]
        assert group[group.valid_sm.gt(0)].spatial_group_id.nunique() == 1
        assert group[group.valid_et.gt(0)].spatial_group_id.nunique() >= (
            1 if split == "train" else 2
        )


def test_insufficient_eval_groups_are_explicit_not_split_or_unlocked():
    from hydrostate.data.site.resplit import task_coverage

    table = samples()
    locked = set(table.spatial_group_id) - {"s1_0"}
    report = task_coverage(table, locked)
    item = next(i for i in report if i["frequency_bin"] == 1 and i["task"] == "sm")
    assert not item["three_way_feasible_by_count"]
    assert item["evaluation_eligible_groups"] == 1
    assignment = stratify_groups(table, locked)
    assert all(assignment[g] == "train" for g in locked)
