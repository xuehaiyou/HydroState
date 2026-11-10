import pandas as pd

from hydrostate.data.site.et_rebalance import rebalance_et


def test_promote_groups_without_training_leakage_or_changing_sm():
    records = []
    for i in range(20):
        split = "train" if i < 6 else ("validation" if i < 13 else "test")
        for day in range(3):
            records.append(
                dict(
                    spatial_group_id=str(i),
                    physical_group_id=str(i),
                    split=split,
                    valid_sm=int(i == 6),
                    valid_et=1,
                    frequency_bin=0,
                    landcover=i % 3,
                    climate_class=i % 2,
                    year=2020,
                    month=day + 1,
                )
            )
    a = pd.DataFrame(records)
    b = rebalance_et(a)
    assert b.split.value_counts().to_dict() == dict(train=42, validation=9, test=9)
    assert b.groupby("spatial_group_id").split.nunique().max() == 1
    assert b.loc[a.split.eq("train"), "split"].eq("train").all()
    assert b.loc[a.valid_sm.gt(0), "split"].eq("validation").all()
    assert b.split.tolist() == rebalance_et(a).split.tolist()


def test_single_rare_evaluation_group_is_retained():
    a = pd.DataFrame(
        [
            dict(
                spatial_group_id="a",
                physical_group_id="a",
                split="train",
                valid_sm=0,
                valid_et=1,
                frequency_bin=1,
                landcover=1,
                climate_class=1,
                year=2020,
                month=1,
            )
        ]
        * 58
        + [
            dict(
                spatial_group_id="b",
                physical_group_id="b",
                split="test",
                valid_sm=0,
                valid_et=1,
                frequency_bin=1,
                landcover=1,
                climate_class=1,
                year=2020,
                month=1,
            )
        ]
        * 8
    )
    assert rebalance_et(a).split.tolist() == a.split.tolist()


def test_additive_failure_refill_resume_and_no_overshoot(tmp_path):
    from hydrostate.data.site.et_rebalance import collect_additive_targets

    base = dict(
        frequency_bin=0,
        landcover=10,
        climate_class=1,
        year=2020,
        month=1,
        target_time="2020-01-01",
        physical_group_id="station",
        valid_et=1,
    )
    selected = pd.DataFrame(
        [
            dict(base, candidate_id=f"{s}-{i}", split=s)
            for s in ("train", "validation", "test")
            for i in range(6)
        ]
    )
    reference = pd.DataFrame([dict(base, candidate_id="old", split="train")])
    path = tmp_path / "reference.parquet"
    reference.to_parquet(path)
    config = dict(
        min_date_gap_days=0,
        seed=43,
        sampling_reference=str(path),
        success_targets_by_split=dict(train=3, validation=1, test=1),
    )
    successes, failures, calls = [], [], []

    def collect(row):
        calls.append(row["candidate_id"])
        return dict(row, status="rejected" if row["candidate_id"].endswith("-0") else "success")

    def consume(result):
        (successes if result["status"] == "success" else failures).append(result)

    assert not collect_additive_targets(
        selected, successes, failures, config, collect, consume, workers=2, max_attempts=2
    )
    assert collect_additive_targets(
        selected, successes, failures, config, collect, consume, workers=4
    )
    assert len(calls) == len(set(calls))
    assert (
        pd.DataFrame(successes).split.value_counts().to_dict() == config["success_targets_by_split"]
    )
    assert collect_additive_targets(selected, successes, failures, config, collect, consume)
    assert len(calls) == len(set(calls))
