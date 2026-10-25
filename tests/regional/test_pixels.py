import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hydrostate.data.gee.client import retry
from hydrostate.data.gee.contracts import atomic_json
from hydrostate.data.gee.state import ExportState
from hydrostate.data.regional import pixels
from hydrostate.data.regional.plan import block_grid, initialize, product
from hydrostate.storage.regional import commit, product_path, ready


def config():
    return json.loads(Path("configs/inference/poyang_pilot.json").read_text())


def test_transport_settings_do_not_change_existing_product_identity(tmp_path):
    cfg = config()
    settings = cfg.pop("pixels")
    first = initialize(tmp_path, cfg)
    frozen = (tmp_path / "plan.json").read_bytes()
    cfg["pixels"] = settings
    cfg["pixels"]["workers"] = 2
    second = initialize(tmp_path, cfg)
    assert first["identity"] == second["identity"]
    assert (tmp_path / "plan.json").read_bytes() == frozen
    assert json.loads((tmp_path / "pixels.json").read_text())["workers"] == 2
    pixels.validate_settings(settings)
    with pytest.raises(ValueError, match="48 MB"):
        pixels.validate_settings(dict(settings, max_request_mib=48))


def test_switch_refuses_unfinished_drive_before_network_access(monkeypatch, tmp_path):
    initialize(tmp_path, config())
    state = ExportState(tmp_path / "exports.sqlite")
    state.add(dict(id="inflight"))
    state.update("inflight", status="submitted", task_id="already_submitted")
    state.close()
    monkeypatch.setattr(pixels, "initialize", lambda *a: pytest.fail("network initialized"))
    with pytest.raises(RuntimeError, match="--backend drive --drain"):
        pixels.prepare(tmp_path)


def test_retry_distinguishes_service_failure_from_invalid_request(monkeypatch):
    monkeypatch.setattr("hydrostate.data.gee.client.time.sleep", lambda *a: None)
    calls = []

    def fail(code):
        calls.append(code)
        error = Exception("service")
        error.resp = SimpleNamespace(status=code)
        raise error

    with pytest.raises(Exception, match="service"):
        retry(lambda: fail(429), attempts=3, retry_if=pixels.transient_error)
    assert calls == [429] * 3
    calls.clear()
    with pytest.raises(Exception, match="service"):
        retry(lambda: fail(400), attempts=3, retry_if=pixels.transient_error)
    assert calls == [400]
    assert not pixels.transient_error(ValueError("Invalid output array"))


def test_partial_resume_empty_source_and_network_failure(monkeypatch, tmp_path):
    import zarr

    cfg = config()
    cfg["pixels"]["workers"] = 1
    plan = initialize(tmp_path, cfg)
    items = [
        product(plan, "s1_10m", slot, [0, 0], [25, "ASCENDING"])
        for slot in plan["periods"][0]["slots"][:3]
    ]
    grid = block_grid(plan, [0, 0])
    h, w = grid["height"], grid["width"]
    commit(tmp_path, plan, items[0], np.zeros((6, h, w), "f4"))
    cached = (product_path(tmp_path, items[0]) / "_SUCCESS.json").stat().st_mtime_ns
    (tmp_path / "targets").mkdir()
    atomic_json(tmp_path / "targets" / (plan["periods"][0]["start"] + ".json"), dict(
        period=plan["periods"][0], plan=plan["identity"], products=items,
    ))
    bands = ["t0_" + b for b in ("VV", "VH", "days", "mean_day", "first_day", "last_day")]

    def jobs(*a, **kw):
        for i, item in enumerate(items):
            if not ready(tmp_path, item):
                yield dict(
                    id=str(i), group="s1_10m", products=[item], grid=grid, bands=bands,
                    scenes=[[] if i == 1 else ["scene"]], raw_bytes=6 * h * w * 4,
                )

    monkeypatch.setattr(pixels, "initialize", lambda *a: None)
    monkeypatch.setattr(pixels, "jobs", jobs)
    monkeypatch.setattr(pixels, "composite_image", lambda ee, job: job)
    network_calls = []

    def failed(*a, **kw):
        network_calls.append("failed")
        raise ValueError("malformed response")

    monkeypatch.setattr(pixels, "compute_pixels", failed)
    with pytest.raises(ValueError, match="malformed response"):
        pixels.prepare(tmp_path)
    assert ready(tmp_path, items[1])
    assert not zarr.open_group(str(product_path(tmp_path, items[1])), mode="r")["valid"][:].any()
    assert not ready(tmp_path, items[2])
    assert network_calls == ["failed"]

    def recovered(*a, **kw):
        network_calls.append("recovered")
        raw = np.zeros((h, w), dtype=[(name, "f4") for name in bands])
        raw["t0_days"] = 1
        return raw

    monkeypatch.setattr(pixels, "compute_pixels", recovered)
    assert pixels.prepare(tmp_path)["ready_dates"] == 1
    assert network_calls == ["failed", "recovered"]
    assert (product_path(tmp_path, items[0]) / "_SUCCESS.json").stat().st_mtime_ns == cached
    assert zarr.open_group(str(product_path(tmp_path, items[2])), mode="r")["valid"][:].all()
    assert json.loads((tmp_path / "pixels_progress.json").read_text())["products"] == 1


def test_request_budget_rejected_before_fetch_or_commit(monkeypatch, tmp_path):
    plan = initialize(tmp_path, config())
    job = dict(group="s1_10m", grid=block_grid(plan, [0, 0]), raw_bytes=33 * 1024**2)
    monkeypatch.setattr(pixels, "compute_pixels", lambda *a, **kw: pytest.fail("network called"))
    with pytest.raises(ValueError, match="byte budget"):
        pixels.receive(None, tmp_path, plan, job, config()["pixels"])
    assert not (tmp_path / "inputs").exists()
