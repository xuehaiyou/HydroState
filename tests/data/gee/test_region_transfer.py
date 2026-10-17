"""Recovery tests for the production Drive queue and the shared scene cache."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
from affine import Affine

from hydrostate.data.gee.backends.drive import IncompleteExport, ingest_export, resume_download
from hydrostate.data.gee.contracts import atomic_json, input_transform
from hydrostate.data.gee.products import BANDS, normalize_pixels
from hydrostate.data.gee.region_plan import export_jobs
from hydrostate.data.gee.state import ExportState
from hydrostate.data.gee.workflows.drive_queue import run_jobs
from hydrostate.storage.scene_cache import SceneCache


@pytest.fixture
def setup(tmp_path):
    plan = dict(
        grid=dict(crs="EPSG:32650", left=400000, top=3200000, height=312, width=312, halo=4),
        config=dict(cache_chunk=480),
    )
    cache = SceneCache(tmp_path, plan)
    scene = dict(key="s1", asset_id="test/s1", bounds=[0, 0, 10000000, 10000000], time=1)
    tr = input_transform(plan["grid"])
    job = dict(
        id="job1",
        scene=scene,
        contract=cache.contract,
        chunk=480,
        grid=plan["grid"],
        bands=[*BANDS["s1"], "valid"],
        chunks=[[0, 0], [0, 1]],
        x=0,
        y=0,
        width=960,
        height=480,
        transform=list(tr)[:6],
        raw_bytes=960 * 480 * 12,
    )
    return cache, scene, job


def tiff(path, job, col=0, width=480, value=2):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=480,
        count=3,
        dtype="float32",
        crs=job["grid"]["crs"],
        transform=Affine(*job["transform"]) * Affine.translation(col, 0),
    ) as dst:
        raw = np.full((3, 480, width), value, dtype="f4")
        raw[-1] = 1
        dst.write(raw)
        dst.descriptions = tuple(job["bands"])
    return path


def test_ingest_requires_all_parts_before_marking_done(setup, tmp_path):
    cache, scene, job = setup
    first = tiff(tmp_path / "a.tif", job)
    with pytest.raises(IncompleteExport):
        ingest_export(cache, job, [first])
    assert not cache.done(scene).any()
    second = tiff(tmp_path / "b.tif", job, col=480)
    ingest_export(cache, job, [first, second])
    assert cache.done(scene).tolist() == [[True, True], [False, False]]
    expected = normalize_pixels(
        "s1", np.full((2, 480, 480), 2, "f4"), np.ones((480, 480), bool), cache.norm
    )
    np.testing.assert_array_equal(cache._open(scene)["values"][:, :480, :480], expected)
    # Re-ingestion must not overwrite chunks already present in the old pixel cache.
    tiff(tmp_path / "a.tif", job, value=9)
    ingest_export(cache, job, [first, second])
    np.testing.assert_array_equal(cache._open(scene)["values"][:, :480, :480], expected)


def test_ingest_rejects_duplicate_and_wrong_grid(setup, tmp_path):
    cache, scene, job = setup
    first = tiff(tmp_path / "a.tif", job)
    with pytest.raises(ValueError, match="duplicate"):
        ingest_export(cache, job, [first, first])
    with rasterio.open(first, "r+") as dst:
        dst.transform = Affine(*job["transform"]) * Affine.translation(0.5, 0)
    with pytest.raises(ValueError, match="not aligned"):
        ingest_export(cache, job, [first])
    assert not cache.done(scene).any()


def test_plan_bounds_bytes_covers_halo_and_skips_done(setup):
    cache, scene, job = setup
    window = dict(row=156, col=156)
    jobs = list(export_jobs(cache, [scene], [window], max_export_bytes=480**2 * 12))
    assert {tuple(c) for j in jobs for c in j["chunks"]} == {(0, 0), (0, 1), (1, 0), (1, 1)}
    assert all(j["raw_bytes"] <= 480**2 * 12 for j in jobs)
    cache.write_chunk(scene, 0, 0, np.zeros((2, 480, 480), "f4"), np.zeros((480, 480), bool))
    remaining = list(export_jobs(cache, [scene], [window], max_export_bytes=480**2 * 12))
    assert len(remaining) == 3
    assert all([0, 0] not in j["chunks"] for j in remaining)


def test_resumes_range_and_retries_truncated_range(tmp_path, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)
    payload = b"0123456789abcdefgh"
    remote = dict(id="blob", size=str(len(payload)), md5Checksum=hashlib.md5(payload).hexdigest())
    path = tmp_path / "scene.tif"
    Path(str(path) + ".part").write_bytes(payload[:4])
    atomic_json(Path(str(path) + ".part.json"), remote)
    requests = []

    class Response:
        status_code = 206

        def __init__(self, start, end, broken):
            self.start, self.end, self.broken = start, end, broken
            self.headers = {"Content-Range": f"bytes {start}-{end}/{len(payload)}"}

        def iter_content(self, **kwargs):
            if self.broken:
                yield payload[self.start : self.start + 2]
                raise OSError("network interrupted")
            yield payload[self.start : self.end + 1]

        def close(self):
            pass

    def get(url, **kwargs):
        start, end = map(int, kwargs["headers"]["Range"].split("=")[1].split("-"))
        requests.append((start, end))
        return Response(start, end, len(requests) == 1)

    resume_download(SimpleNamespace(get=get), remote, path, chunk_bytes=4)
    assert requests[:2] == [(4, 7), (4, 7)]
    assert path.read_bytes() == payload
    assert not Path(str(path) + ".part").exists()


def test_checksum_failure_never_publishes_file(tmp_path):
    remote = dict(id="blob", size="3", md5Checksum=hashlib.md5(b"abc").hexdigest())
    response = SimpleNamespace(
        status_code=206,
        headers={"Content-Range": "bytes 0-2/3"},
        iter_content=lambda **_: iter([b"bad"]),
        close=lambda: None,
    )
    with pytest.raises(ValueError, match="checksum"):
        resume_download(SimpleNamespace(get=lambda *a, **k: response), remote, tmp_path / "x")
    assert not (tmp_path / "x").exists()


def test_cleanup_requires_commit_and_recovers_without_resubmission(setup, tmp_path):
    cache, scene, job = setup
    settings = dict(
        max_exports=2,
        download_workers=2,
        file_dimensions=3840,
        max_export_gib=0.5,
        max_pending_gib=1.0,
        drive_reserve_gib=0.5,
        poll_seconds=0.001,
    )
    state = ExportState(tmp_path / "exports.sqlite")
    state.add(job)
    with pytest.raises(RuntimeError, match="committed"):
        state.update(job["id"], status="cleaned")
    state.update(job["id"], status="submitted", task_id="already-submitted")
    files = [
        dict(id="a", name="a.tif", size="1", md5Checksum="one"),
        dict(id="b", name="b.tif", size="1", md5Checksum="two"),
    ]
    paths = [tiff(tmp_path / "a.tif", job), tiff(tmp_path / "b.tif", job, col=480)]
    calls = []

    def cleanup(record):
        assert cache.done(scene)[0].all()
        assert record["status"] == "committed"
        calls.append("cleaned")

    transport = SimpleNamespace(
        ensure_folder=lambda: None,
        status=lambda _: {"state": "COMPLETED"},
        inventory=lambda _: files,
        fetch=lambda *_: paths,
        cleanup=cleanup,
        submit=lambda *_: pytest.fail("Duplicate submission"),
    )
    run_jobs(transport, state, cache, [], settings)
    assert state.get(job["id"])["status"] == "cleaned"
    assert calls == ["cleaned"]
    state.close()


def test_commit_survives_cleanup_interruption(setup, tmp_path):
    cache, scene, job = setup
    ingest_export(
        cache, job, [tiff(tmp_path / "a.tif", job), tiff(tmp_path / "b.tif", job, col=480)]
    )
    state = ExportState(tmp_path / "exports.sqlite")
    state.add(job)
    state.update(job["id"], status="committed", files=[])
    state.close()
    state = ExportState(tmp_path / "exports.sqlite")
    settings = json.loads(Path("configs/data/gee_drive.json").read_text())
    calls = []
    transport = SimpleNamespace(ensure_folder=lambda: None, cleanup=lambda r: calls.append(r["id"]))
    run_jobs(transport, state, cache, [], settings)
    assert calls == ["job1"]
    assert not state.pending()
    state.close()


def test_sample_pixels_use_verified_named_bands_and_resume(tmp_path, monkeypatch):
    from hydrostate.data.gee.workflows import samples

    row = dict(
        sample_id="sample",
        inputs="{}",
        labels="{}",
        transform=json.dumps([10, 0, 400000, 0, -10, 3200000]),
        crs="EPSG:32650",
    )
    names = ["one", "two"]
    calls = []

    def compute(request):
        calls.append(request)
        raw = np.zeros((120, 120), dtype=[(name, "f4") for name in names])
        raw["one"] = 3
        return raw

    def ingest(path, sample, bands, root):
        with rasterio.open(path) as src:
            assert list(src.descriptions) == names
            assert src.read(1).mean() == 3
            assert src.transform.c == 400000
        return "inputs.zarr", "labels.zarr"

    monkeypatch.setattr(samples, "export_image", lambda *_: (object(), names))
    monkeypatch.setattr(samples, "ingest", ingest)
    ee = SimpleNamespace(data=SimpleNamespace(computePixels=compute))
    args = SimpleNamespace(concurrency=1, batch_size=1)
    samples.run_pixels(ee, args, tmp_path, [row])
    samples.run_pixels(ee, args, tmp_path, [row])
    assert len(calls) == 1
    assert not (tmp_path / "downloads").exists()


def test_capacity_failure_preserves_planned_receipt_without_submission(setup, tmp_path):
    cache, scene, job = setup
    settings = json.loads(Path("configs/data/gee_drive.json").read_text())
    state = ExportState(tmp_path / "exports.sqlite")
    transport = SimpleNamespace(
        ensure_folder=lambda: None,
        available_bytes=lambda: 0,
        submit=lambda *_: pytest.fail("Exceeded Drive quota"),
    )
    with pytest.raises(OSError, match="capacity"):
        run_jobs(transport, state, cache, [job], settings)
    assert state.get(job["id"])["status"] == "planned"
    state.close()
