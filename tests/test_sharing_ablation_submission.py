"""Three-way experimental isolation, submission persistence and paired analysis."""

import json
import subprocess

import numpy as np
import pandas as pd
import pytest
from mmengine.config import Config

from hydrostate.cli import configure_data_contract
from tools.legacy import dispatch_sharing_ablation as dispatch
from tools.legacy import submit_sharing_ablation as submit
from tools.legacy.summarize_sharing_ablation import paired_summary, summarize


def test_three_way_configs_and_seed_grid(monkeypatch):
    for key in list(submit.os.environ):
        if key.startswith("HYDROSTATE_"):
            monkeypatch.delenv(key)
    plan = submit.experiment_plan([3407, 3408, 3409])
    assert len(plan) == 15
    paths = []
    for entry in plan:
        monkeypatch.setenv("HYDROSTATE_ABLATION_SEED", str(entry["seed"]))
        monkeypatch.setenv("HYDROSTATE_ABLATION_TASK", entry["task"] or "soil_moisture")
        cfg = Config.fromfile(entry["config"])
        submit.validate_config(cfg, entry)
        paths.append(cfg.work_dir)
    assert len(set(paths)) == 15
    with pytest.raises(ValueError, match="unique"):
        submit.experiment_plan([1, 1])


def test_submit_all_training_jobs_once_and_summary_depends_on_every_job(tmp_path, monkeypatch):
    root = tmp_path / "run"
    model = tmp_path / "model"
    model.mkdir()
    for name in ("config.json", "weights.pth"):
        (model / name).write_text("fixture")
    pack = tmp_path / "inputs/landsat_l2_sr_v1"
    pack.mkdir(parents=True)
    for name in ("_SUCCESS.json", "metadata.json", "index.parquet"):
        (pack / name).write_text("fixture")
    manifest, stats = tmp_path / "manifest.parquet", tmp_path / "stats.json"
    manifest.write_text("fixture")
    stats.write_text("{}")
    monkeypatch.setattr("sys.argv", ["submit", "--root", str(root), "--data-root", str(tmp_path),
                                    "--manifest", str(manifest), "--statistics", str(stats),
                                    "--model-path", str(model)])
    monkeypatch.setattr(submit.os, "environ", dict(submit.os.environ))
    monkeypatch.setattr(submit, "read_manifest", lambda *args, **kwargs: pd.DataFrame(
        dict(sample_id=["a", "b", "c"], split=["train", "validation", "test"],
             lon=[1., 11., 21.], lat=[1., 1., 1.])))

    def statistics(cfg, training):
        configure_data_contract(cfg)
        cfg.model.target_statistics = dict(training_fingerprint="0" * 64)

    monkeypatch.setattr(submit, "configure_target_statistics", statistics)
    monkeypatch.setattr(submit.subprocess, "check_output", lambda args, **kwargs:
                        "a" * 40 if "rev-parse" in args else b"tools/train.py\0")
    calls = []

    def run(command, **kwargs):
        if command[0] == "git":
            return subprocess.CompletedProcess(command, 0)
        record = json.loads((root / "submission.json").read_text())
        assert sum(j["status"] == "submitted" for j in record["jobs"].values()) == len(calls)
        assert sum(j["status"] == "submitting" for j in record["jobs"].values()) == 1
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, f"{100 + len(calls)}\n", "")

    monkeypatch.setattr(submit.subprocess, "run", run)
    submit.main()
    assert len(calls) == 16
    assert not any("dependency" in arg for c in calls[:-1] for arg in c)
    assert "--dependency=afterok:" + ":".join(map(str, range(101, 116))) in calls[-1]
    record = json.loads((root / "submission.json").read_text())
    for entry in record["experiments"]:
        cfg = Config.fromfile(entry["expanded_config"])
        submit.validate_config(cfg, entry)
        assert cfg.model.target_statistics.training_fingerprint == "0" * 64
        assert submit.sha256(entry["expanded_config"]) == entry["config_sha256"]
    with pytest.raises(FileExistsError):
        submit.main()
    assert len(calls) == 16


def test_paired_spatial_summary_has_correct_direction_and_rejects_support_drift():
    cube = np.zeros((3, 3, 4, 3, 2))
    cube[..., 1] = 10
    for group, rmse in enumerate([3., 1., 2.]):
        cube[group, ..., 0] = 10 * rmse ** 2
    result = paired_summary(cube, np.ones(3), draws=50)
    assert result["groups"]["shared_private"]["rmse_mean"] == [1., 1., 1.]
    assert result["groups"]["shared_private"]["gain_vs_independent_percent"] == [50., 50., 50.]
    intervals = result["paired_differences"]["shared_private_minus_independent"]["ci95"]
    assert intervals == [[-1., -1.]] * 4
    cube[0, 0, 0, 0, 1] += 1
    with pytest.raises(ValueError, match="supports differ"):
        paired_summary(cube, np.ones(3), draws=50)


def test_summary_joins_independent_tasks_and_checks_sample_identity(tmp_path):
    manifest, statistics = tmp_path / "manifest.parquet", tmp_path / "stats.json"
    pd.DataFrame(dict(sample_id=["a", "b"], split=["test", "test"],
                      lon=[1., 11.], lat=[1., 1.])).to_parquet(manifest)
    statistics.write_text(json.dumps(dict(tasks={t: dict(std=1.) for t in submit.TASKS})))
    plan = submit.experiment_plan([1, 2, 3])
    for entry in plan:
        directory = tmp_path / entry["stage"] / "evaluation"
        directory.mkdir(parents=True)
        tasks = [entry["task"]] if entry["task"] else list(submit.TASKS)
        entry.update(work_dir=str(directory.parent), selected_tasks=tasks, config_sha256="a" * 64)
        rmse = dict(shared=3., shared_private=1., independent=2.)[entry["group"]]
        samples = directory / "samples.jsonl"
        samples.write_text("".join(json.dumps(dict(sample_id=sample_id, **{
            key: value for t in tasks for key, value in
            [(f"{t}/sum_sq", rmse ** 2), (f"{t}/count", 1)]})) + "\n"
                                   for sample_id in ("a", "b")))
        result = dict(sample_metrics=str(samples), parameters_by_owner={},
                      metrics={f"{t}_rmse": rmse for t in tasks})
        (directory / "metrics.json").write_text(json.dumps(dict(
            seed=entry["seed"], config_sha256=entry["config_sha256"], selected_tasks=tasks,
            training_fingerprint="f" * 64, training_wall_seconds=12,
            results=dict(final=result, best=result))))
    receipt = tmp_path / "submission.json"
    receipt.write_text(json.dumps(dict(
        manifest=str(manifest), manifest_sha256=submit.sha256(manifest),
        statistics=str(statistics), statistics_sha256=submit.sha256(statistics),
        commit="a" * 40, training_fingerprint="f" * 64, seeds=[1, 2, 3], experiments=plan)))
    report = summarize(receipt, draws=25)
    assert report["evaluations"]["final"]["groups"]["independent"]["rmse_mean"] == [2.] * 3
    assert len(report["runs"]) == 15
    # A different test cohort cannot silently enter a paired comparison.
    samples.write_text(samples.read_text().replace('"b"', '"wrong-id"'))
    with pytest.raises(ValueError, match="sample identities"):
        summarize(receipt, draws=25)


def test_quota_controller_only_submits_missing_jobs_and_has_one_successor(tmp_path, monkeypatch):
    path = tmp_path / "submission.json"
    record = dict(code_root="/code", run_root=str(tmp_path), environment={}, jobs={},
                  experiments=submit.experiment_plan([3407]), gpu_submission_limits={})
    path.write_text(json.dumps(record))
    remaining = [1]

    def capacity(record):
        return dict(free=remaining[0], job_ids=["existing"])

    monkeypatch.setattr(dispatch, "capacity", capacity)
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        if command[-2].endswith("35_train_sharing_ablation.sh"):
            remaining[0] -= 1
        return subprocess.CompletedProcess(command, 0, f"{100 + len(commands)}\n", "")

    monkeypatch.setattr(dispatch.subprocess, "run", run)
    dispatch.dispatch_pending(path, {})
    record = json.loads(path.read_text())
    assert record["jobs"]["shared_s3407"]["job_id"] == "101"
    assert record["active_dispatcher"] == "102"
    assert len(record["waiting_for_gpu_submission"]) == 4
    dispatch.dispatch_pending(path, {})
    assert len(commands) == 2  # A second interactive invocation cannot duplicate the chain.
    monkeypatch.setenv("SLURM_JOB_ID", "102")
    remaining[0] = 10
    dispatch.dispatch_pending(path, {})
    record = json.loads(path.read_text())
    assert record["waiting_for_gpu_submission"] == []
    assert "active_dispatcher" not in record
    assert "--dependency=afterok:101:103:104:105:106" in commands[-1]
    assert len(commands) == 7


def test_quota_race_is_not_retried_until_queue_changes(tmp_path, monkeypatch):
    path = tmp_path / "submission.json"
    path.write_text(json.dumps(dict(
        code_root="/code", run_root=str(tmp_path), environment={}, jobs={},
        experiments=submit.experiment_plan([3407]))))
    monkeypatch.setattr(dispatch, "capacity", lambda record: dict(free=1, job_ids=["unchanged"]))
    attempts = []

    def run(command, **kwargs):
        attempts.append(command)
        if command[-2].endswith("35_train_sharing_ablation.sh"):
            return subprocess.CompletedProcess(command, 1, "", "MaxSubmitJobsPerAccount")
        return subprocess.CompletedProcess(command, 0, f"{100 + len(attempts)}\n", "")

    monkeypatch.setattr(dispatch.subprocess, "run", run)
    dispatch.dispatch_pending(path, {})
    record = json.loads(path.read_text())
    monkeypatch.setenv("SLURM_JOB_ID", record["active_dispatcher"])
    dispatch.dispatch_pending(path, {})
    assert len(attempts) == 3  # One rejected GPU submission, two controller submissions.
    assert "--begin=now+10minutes" in attempts[-1]
