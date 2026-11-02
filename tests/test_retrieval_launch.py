import json
from types import SimpleNamespace

import pytest

from hydrostate.experiments import retrieval_launch as launch


def test_epoch_budget_keeps_margin_and_splits_resumable_work():
    assert launch.training_segments(300, 50) == [
        dict(max_epochs=50, seconds=16500, walltime="04:35:00")]
    parts = launch.training_segments(3600, 5, max_seconds=8000)
    assert [p["max_epochs"] for p in parts] == [2, 4, 5]
    assert all(p["seconds"] <= 8000 for p in parts)
    with pytest.raises(ValueError, match="queue limit"):
        launch.training_segments(10000, 50, max_seconds=1000)


@pytest.fixture
def run(tmp_path):
    debug = tmp_path / "debug"
    scalar = debug / "timestamp" / "vis_data" / "scalars.json"
    scalar.parent.mkdir(parents=True)
    scalar.write_text(json.dumps(dict(loss=1.2, memory=5000)) + "\n"
                      + json.dumps(dict(hydro_score=-0.5)) + "\n")
    (debug / "epoch_1.pth").write_bytes(b"checkpoint fixture")
    record = dict(
        source_release=str(tmp_path / "release"), source_hashes=str(tmp_path / "source.sha256"),
        environment=dict(HYDROSTATE_WORK_DIR=str(tmp_path / "unet")),
        gpu_memory_mib=48000, regular_qos="gpu", regular_max_seconds=604800,
        jobs=[dict(stage="debug", job_id="123", work_dir=str(debug), microbatch=16,
                   accumulation=1, effective_batch=16, command=[
                       "sbatch", "--parsable", "--job-name=debug", "--qos=debug", "--time=00:25:00",
                       "--output=debug_%j.out", "--error=debug_%j.err", "--partition=l40s",
                       str(tmp_path / "release/scripts/slurm/train_retrieval.sh"), "unet",
                       "train_cfg.max_epochs=1", "train_dataloader.batch_size=16",
                       "optim_wrapper.accumulative_counts=1", "param_scheduler.0.end=125",
                   ])],
    )
    (tmp_path / "submission.json").write_text(json.dumps(record))
    return tmp_path, record, scalar


def test_debug_success_requires_finite_loss_validation_and_memory(run, monkeypatch):
    path, record, scalar = run
    monkeypatch.setattr(launch.subprocess, "check_output",
                        lambda *a, **k: "123|COMPLETED|0:0|300|\n")
    assert launch.inspect_debug(path, record["jobs"][0], 48000)["epoch_seconds"] == 300
    scalar.write_text('{"loss":NaN,"memory":5000}\n{"hydro_score":-1}\n')
    with pytest.raises(ValueError, match="Nonfinite training"):
        launch.inspect_debug(path, record["jobs"][0], 48000)
    scalar.write_text('{"loss":1,"memory":47000}\n{"hydro_score":-1}\n')
    with pytest.raises(ValueError, match="headroom"):
        launch.inspect_debug(path, record["jobs"][0], 48000)
    monkeypatch.setattr(launch.subprocess, "check_output", lambda *a, **k: "123|FAILED|1:0|300|\n")
    with pytest.raises(ValueError, match="did not complete"):
        launch.inspect_debug(path, record["jobs"][0], 48000)


def test_formal_submission_reuses_recipe_starts_fresh_and_is_idempotent(run, monkeypatch):
    path, record, _ = run
    monkeypatch.setattr(launch.subprocess, "check_output",
                        lambda *a, **k: "123|COMPLETED|0:0|300|\n")
    calls = []

    def execute(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="456;hpc\n", stderr="")

    monkeypatch.setattr(launch.subprocess, "run", execute)
    result = launch.submit_after_debug(path)
    command = calls[-1]
    assert "--time=04:35:00" in command and "--qos=gpu" in command
    assert "train_cfg.max_epochs=50" in command and "resume=False" in command
    assert "param_scheduler.0.end=125" in command and "--dependency=afterok:123" in command
    assert "--kill-on-invalid-dep=yes" in command
    assert command.index("--dependency=afterok:123") < command.index("unet")
    assert result["jobs"][-1]["job_id"] == "456"
    assert result["jobs"][-1]["cluster"] == "hpc"
    launch.submit_after_debug(path)
    assert len(calls) == 2  # One checksum check and exactly one sbatch.
    record["status"] = "prepared"
    (path / "submission.json").write_text(json.dumps(record))
    with pytest.raises(FileExistsError):
        launch.submit_after_debug(path)


def test_later_segment_resumes_the_same_model(run):
    path, record, _ = run
    command = launch.formal_command(record, record["jobs"][0],
                                    dict(max_epochs=50, walltime="01:00:00"), 1, "456", path)
    assert "resume=True" in command and "--dependency=afterok:456" in command
