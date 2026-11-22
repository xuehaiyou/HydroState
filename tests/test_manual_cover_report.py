"""Keep checkpoint selection separate from individual task optima and final test."""

import json

import pytest

from hydrostate.experiments.manual_cover_report import compare_validation, summarize


def write_rows(root, part, rows):
    path = root / "CGC" / part / "20261122" / "vis_data" / "scalars.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def test_reports_same_selected_epoch_and_separate_test(tmp_path):
    rows = [
        dict(step=1, manual_joint_score=-0.5, **{"site/site_sm_rmse": 0.1}),
        dict(step=2, manual_joint_score=-0.4, **{"site/site_sm_rmse": 0.2}),
        dict(step=6, manual_joint_score=-0.4, **{"site/site_sm_rmse": 0.05}),
    ]
    path = write_rows(tmp_path, "work", rows)
    with path.open("a") as stream:
        stream.write('{"incomplete":')
    checkpoint = tmp_path / "CGC/work/best_manual_joint_score_epoch_2.pth"
    checkpoint.touch()
    write_rows(
        tmp_path, "final_test", [dict(manual_joint_score=-0.6, **{"site/site_sm_rmse": 0.3})]
    )
    report = summarize(tmp_path)["models"]
    assert report["CGC"]["selected_epoch"] == 2
    assert report["CGC"]["validation"]["site/site_sm_rmse"] == 0.2
    assert report["CGC"]["test"]["site/site_sm_rmse"] == 0.3
    assert report["CGC"]["status"] == "metrics_available"
    assert report["B"]["status"] == "no_validation_result"


def test_refuses_ambiguous_test_results(tmp_path):
    row = dict(step=6, manual_joint_score=-0.5)
    write_rows(tmp_path, "work", [row])
    write_rows(tmp_path, "final_test", [row, row])
    with pytest.raises(ValueError, match="Multiple final-test"):
        summarize(tmp_path)
    assert summarize(tmp_path, validation_only=True)["models"]["CGC"]["test"] is None


def test_joint_lead_requires_each_source_and_shared_support():
    row = {
        "cover/glacier/all/accuracy": 0.8,
        "cover/sen1floods11/all/water_iou": 0.72,
        "site/site_sm_correlation": 0.6,
        "site/site_et_correlation": 0.84,
        "cover/sen1floods11/all/water_mae": 0.11,
        "site/site_sm_rmse": 0.09,
        "site/site_et_rmse": 1.0,
        "site/site_sm_count": 128,
        "site/site_et_count": 64,
        "cover/sen1floods11/all/water_count": 600,
    }
    for c in ("water", "snow_ice", "land"):
        row["cover/glacier/all/" + c + "_mae"] = 0.12
        row["cover/glacier/all/" + c + "_count"] = 40
    other = dict(row, **{"site/site_et_rmse": 0.95, "cover/sen1floods11/all/water_mae": 0.13})
    models = {
        k: dict(validation=v, completed_validation_epoch=6, checkpoint_present=True)
        for k, v in (("CGC", row), ("candidate", other))
    }
    report = compare_validation(models)
    assert report["leaders"]["et_rmse"] == ["candidate"]
    assert report["leaders"]["sm_rmse"] == ["CGC", "candidate"]
    assert report["best_on_four_errors"] == []
    other["cover/sen1floods11/all/water_mae"] = 0.10
    other["site/site_sm_correlation"] = 0.59
    report = compare_validation(models)
    assert report["best_on_four_errors"] == ["candidate"]
    assert report["best_on_all_compared"] == []
    other["site/site_et_count"] -= 1
    assert compare_validation(models)["status"] == "inspect_validation_support"
