"""Submission records and dependencies prevent accidental duplicate GPU jobs."""

import json
import subprocess

import pytest

from tools import submit_shared_private as submit


def test_two_experiment_graph_records_every_job_before_next_submission(tmp_path, monkeypatch):
    root = tmp_path / "run"
    model = tmp_path / "pretrained"
    model.mkdir()
    for name in ("config.json", "weights.pth"):
        (model / name).write_text("fixture")
    manifest, stats = tmp_path / "samples.parquet", tmp_path / "stats.json"
    manifest.write_text("fixture")
    stats.write_text("{}")
    monkeypatch.setattr("sys.argv", ["submit", "--root", str(root),
                                    "--data-root", str(tmp_path), "--manifest", str(manifest),
                                    "--statistics", str(stats), "--model-path", str(model)])
    for key in list(submit.os.environ):
        if key.startswith("HYDROSTATE_"):
            monkeypatch.delenv(key)
    # Restore environment after the submitter sets its run-specific values.
    with monkeypatch.context() as environment:
        environment.setattr(submit.os, "environ", dict(submit.os.environ))
        monkeypatch.setattr(submit, "configure_target_statistics", lambda cfg, training:
                            cfg.model.update(target_statistics=dict(training_fingerprint="0" * 64)))
        monkeypatch.setattr(submit.subprocess, "check_output", lambda args, **kwargs:
                            "a" * 40 if "rev-parse" in args else b"tools/train.py\0")
        calls = []

        def run(command, **kwargs):
            if command[0] == "git":
                return subprocess.CompletedProcess(command, 0)
            record = json.loads((root / "submission.json").read_text())
            assert sum(v["status"] == "submitting" for v in record["jobs"].values()) == 1
            assert sum(v["status"] == "submitted" for v in record["jobs"].values()) == len(calls)
            calls.append(command)
            return subprocess.CompletedProcess(command, 0, f"{100 + len(calls)};hpc2021\n", "")

        monkeypatch.setattr(submit.subprocess, "run", run)
        submit.main()
        record = json.loads((root / "submission.json").read_text())
        assert list(record["jobs"]) == ["joint50", "shared30", "water", "soil_moisture", "et",
                                        "compose"]
        assert all(v["status"] == "submitted" for v in record["jobs"].values())
        assert not any("dependency" in arg for c in calls[:2] for arg in c)
        assert all("--dependency=afterok:102" in c for c in calls[2:5])
        assert "--dependency=afterok:103:104:105" in calls[5]
        assert record["experiments"]["joint"]["epochs"] == 50
        assert record["experiments"]["staged"] == dict(shared_epochs=30, private_epochs=20)
        with pytest.raises(FileExistsError, match="Existing run"):
            submit.main()
        assert len(calls) == 6


def test_unrecognized_sbatch_response_keeps_intent_and_prevents_retry(tmp_path, monkeypatch):
    record = dict(jobs={})
    path = tmp_path / "submission.json"
    monkeypatch.setattr(submit.subprocess, "run", lambda command, **kwargs:
                        subprocess.CompletedProcess(command, 0, "ambiguous response", ""))
    with pytest.raises(RuntimeError, match="Check Slurm"):
        submit.submit_stage(record, path, "shared30", ["sbatch"], {})
    assert json.loads(path.read_text())["jobs"]["shared30"]["stdout"] == "ambiguous response"
    with pytest.raises(ValueError, match="Already recorded"):
        submit.submit_stage(record, path, "shared30", ["sbatch"], {})


def test_best_checkpoint_requires_one_metric_match(tmp_path):
    with pytest.raises(ValueError, match="got 0"):
        submit.best_checkpoint(tmp_path, "hydro_score")
    expected = tmp_path / "best_hydro_score_epoch_7.pth"
    expected.touch()
    (tmp_path / "epoch_30.pth").touch()
    assert submit.best_checkpoint(tmp_path, "hydro_score") == expected
    (tmp_path / "best_hydro_score_epoch_8.pth").touch()
    with pytest.raises(ValueError, match="got 2"):
        submit.best_checkpoint(tmp_path, "hydro_score")
