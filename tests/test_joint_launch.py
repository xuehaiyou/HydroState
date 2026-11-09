import json
from types import SimpleNamespace

import pytest

from hydrostate.experiments import joint_launch as launch


@pytest.fixture
def run(tmp_path):
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    write(tmp_path / "debug_passed.json", dict(status="passed", micro_batch=16, accumulation=1))
    write(tmp_path / "preflight.json", dict(status="passed", site_score_weight=2 / 3,
          splits=dict(test=dict(products=20, sites=30))))
    for variant in ("b", "c"):
        write(tmp_path / f"benchmark_{variant}.json", dict(
            status="passed", batch_size=16, train_seconds_per_sample=.1,
            validation_seconds_per_sample=.05, peak_allocated_gib=30,
            peak_reserved_gib=32, gpu_total_gib=48, gpu="L40", train_samples=100,
            validation_samples=50))
        write(tmp_path / f"smoke_{variant}" / "result.json", dict(
            status="passed", micro_batch=16, model="CoSAHydroStateModel",
            site_loss_weights=dict(site_sm=2., site_et=2.), peak_allocated_gib=31,
            peak_reserved_gib=33, elapsed_seconds=20,
            selection={name: dict(samples=16) for name in ("train", "val", "test")}))
    write(tmp_path / "submission.json", dict(
        source_release=str(tmp_path / "release"),
        resources=dict(account="geog_geors", partition="l40s", gres="gpu:nvidia_l40s:1",
                       regular_qos="gpu", regular_max_seconds=7 * 86400,
                       cpus_per_task=4, memory="48G"),
        environment=dict(HYDROSTATE_SOURCE_HASHES=str(tmp_path / "source.sha256"),
                         HYDROSTATE_SLURM_PARTITION="l40s"),
        jobs=[dict(stage="debug", job_id="123", command=[
            "sbatch", "--partition=l40s", "--gres=gpu:nvidia_l40s:1",
            "--cpus-per-task=4", "--mem=48G"])]))
    return tmp_path


def test_measurement_requires_matched_batch_priority_and_headroom(run):
    passed, reports = launch.measurements(run, 50)
    assert passed["micro_batch"] == 16
    assert reports["b"]["fixed_overhead_seconds"] == pytest.approx(16.8)
    assert reports["b"]["epoch_seconds"] == pytest.approx(29.3)
    path = run / "benchmark_c.json"
    value = json.loads(path.read_text())
    value["peak_reserved_gib"] = 46
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="headroom"):
        launch.measurements(run, 50)
    value["peak_reserved_gib"] = 32
    value["batch_size"] = 8
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="recipe"):
        launch.measurements(run, 50)


def test_submission_is_paired_fresh_guarded_and_idempotent(run, monkeypatch):
    monkeypatch.setattr(launch.subprocess, "check_output", lambda *a, **kw: "123|COMPLETED|0:0|\n")
    calls = []

    def execute(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout="456\n", stderr="",
                               check_returncode=lambda: None)

    monkeypatch.setattr(launch.subprocess, "run", execute)
    result = launch.submit_after_debug(run)
    assert result["status"] == "formal_submitted"
    assert len(calls) == 3  # One checksum, exactly two submissions.
    assert "--partition=l40s" in calls[1][0]
    assert "--gres=gpu:nvidia_l40s:1" in calls[1][0]
    assert "--cpus-per-task=4" in calls[1][0]
    for variant, (_, args) in zip(("b", "c"), calls[1:]):
        env = args["env"]
        assert env["HYDROSTATE_MICRO_BATCH"] == "16"
        assert env["HYDROSTATE_ACCUMULATIVE_COUNTS"] == "1"
        script = (run / f"formal_{variant}_1.sh").read_text()
        assert "resume=False load_from=None" in script
        assert "tools/test.py" in script and "best_joint_score_*.pth" in script
    assert calls[1][1]["env"]["HYDROSTATE_CONFIG"].endswith("joint_independent.py")
    assert calls[2][1]["env"]["HYDROSTATE_CONFIG"].endswith("hydrostate_joint.py")
    launch.submit_after_debug(run)
    assert len(calls) == 3


def test_failed_debug_never_submits(run, monkeypatch):
    monkeypatch.setattr(launch.subprocess, "check_output", lambda *a, **kw: "123|FAILED|1:0|\n")
    monkeypatch.setattr(launch.subprocess, "run", lambda *a, **kw: pytest.fail("unexpected submit"))
    with pytest.raises(ValueError, match="not completed"):
        launch.submit_after_debug(run)
    assert not (run / "formal.launch.lock").exists()


def test_resource_mismatch_never_submits(run, monkeypatch):
    path = run / "submission.json"
    record = json.loads(path.read_text())
    record["resources"]["partition"] = "c_foss_gpu"
    path.write_text(json.dumps(record))
    monkeypatch.setattr(launch.subprocess, "run", lambda *a, **kw: pytest.fail("unexpected submit"))
    with pytest.raises(ValueError, match="disagree"):
        launch.submit_after_debug(run)
    record["resources"]["partition"] = "l40s"
    record["resources"]["gres"] = "gpu:nvidia_l40:1"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="debug allocation: gres"):
        launch.submit_after_debug(run)


def test_only_last_segment_runs_test_and_later_segments_resume():
    from pathlib import Path

    earlier = launch.job_script(Path("/code"), "config.py", Path("/work"), 10,
                                resume=False, final=False, batch=16)
    later = launch.job_script(Path("/code"), "config.py", Path("/work"), 50,
                              resume=True, final=True, batch=16)
    assert "tools/test.py" not in earlier
    assert "resume=True" in later and "tools/test.py" in later
