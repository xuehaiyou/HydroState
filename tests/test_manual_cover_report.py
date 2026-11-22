"""Keep checkpoint selection separate from individual task optima and final test."""

import json

import pytest

from hydrostate.experiments.manual_cover_report import summarize


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
