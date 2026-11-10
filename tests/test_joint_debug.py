import json
from pathlib import Path

import pytest

from hydrostate.experiments import joint_debug as debug


def test_extended_check_falls_back_to_next_verified_batch_for_both_variants(tmp_path, monkeypatch):
    calls = []

    def execute(root, args, name, report, **kwargs):
        calls.append(name)
        value = dict(status="passed", peak_allocated_gib=30, peak_external_gib=1,
                     gpu_total_gib=48, memory_samples=8)
        if name == "probe_c_32":
            value = dict(status="oom")
        if name == "smoke_c_28":
            value["peak_allocated_gib"] = 44
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(value))
        return value

    monkeypatch.setattr(debug, "execute", execute)
    result = debug.run(tmp_path, probe_batches=(24, 28, 32, 36))
    assert result["micro_batch"] == 24
    assert "probe_c_36" not in calls
    for variant in ("b", "c"):
        assert f"benchmark_{variant}_24" in calls
        assert f"smoke_{variant}_24" in calls
        expected = tmp_path / "batch_24" / f"smoke_{variant}"
        assert (tmp_path / f"smoke_{variant}").resolve() == expected
    assert (tmp_path / "batch_28" / "smoke_c" / "result.json").is_file()
    with pytest.raises(FileExistsError):
        debug.run(tmp_path, probe_batches=(24, 28, 32))


def test_non_memory_failure_is_not_hidden_by_lower_batch_retry(tmp_path, monkeypatch):
    def execute(root, args, name, report, **kwargs):
        if name == "preflight":
            return dict(status="passed")
        raise RuntimeError("Nonfinite joint training loss")

    monkeypatch.setattr(debug, "execute", execute)
    with pytest.raises(RuntimeError, match="Nonfinite"):
        debug.run(tmp_path, probe_batches=(24, 28))
    assert not (Path(tmp_path) / "debug_passed.json").exists()
