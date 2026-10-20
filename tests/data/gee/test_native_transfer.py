"""Native grids, QA parity, missing-file protection and multipart recovery."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
from affine import Affine

from hydrostate.data.gee.backends.drive import IncompleteExport, ingest_export
from hydrostate.data.gee.contracts import input_transform
from hydrostate.data.gee.native import NativeReader, band_groups, native_job
from hydrostate.data.gee.products import BANDS, normalize_pixels
from hydrostate.data.gee.region_plan import selected_export_jobs
from hydrostate.data.gee.state import ExportState
from hydrostate.data.gee.workflows.drive_queue import run_jobs, validate_settings
from hydrostate.storage.scene_cache import SceneCache


def source_info(key, origin):
    bands = []
    for name in [*BANDS[key], *{"s1": [], "s2": ["SCL"],
                              "landsat": ["QA_PIXEL", "QA_RADSAT"]}[key]]:
        if key == "s2":
            scale = 10 if name in ["B2", "B3", "B4", "B8"] else (
                60 if name in ["B1", "B9"] else 20
            )
        else:
            scale = 10 if key == "s1" else (15 if name == "B8" else 30)
        bands.append(dict(
            id=name, crs="EPSG:32650",
            crs_transform=[scale, 0, origin[0], 0, -scale, origin[1]],
            data_type=dict(precision="double" if key == "s1" else "int", min=0, max=65535),
        ))
    return dict(bands=bands)


@pytest.fixture
def native_setup(tmp_path):
    def make(key):
        root = tmp_path / key
        root.mkdir()
        plan = dict(grid=dict(crs="EPSG:32650", left=400000, top=3200000,
                              height=32, width=32, halo=4), config=dict(cache_chunk=120))
        cache = SceneCache(root, plan)
        scene = dict(key=key, asset_id="test/" + key)
        tr = input_transform(cache.grid)
        groups = band_groups(scene, source_info(key, (tr.c-5, tr.f+5)))
        demand = [dict(scene=scene, chunks={(0, 0)})]
        job = next(selected_export_jobs(cache, demand, max_export_bytes=1e8))
        return cache, scene, native_job(job, groups)
    return make


def write_parts(root, job, *, split=False, invalid=False, border=(0, 0, 0, 0),
                coordinates=False):
    paths = {}
    left, top, right, bottom = border
    for part in job["parts"]:
        paths[part["id"]] = []
        cuts = [-left, part["width"] + right]
        if split:
            cuts.insert(1, part["width"] // 2)
        for start, end in zip(cuts, cuts[1:]):
            path = root / f"{part['id']}_{start}.tif"
            height = part["height"] + top + bottom
            values = np.full((len(part["bands"]), height, end-start),
                             100, part["dtype"])
            for i, name in enumerate(part["bands"]):
                if coordinates and name in BANDS[job["scene"]["key"]]:
                    yy, xx = np.indices((height, end-start))
                    yy, xx = yy-top, xx+start
                    inside = (yy >= 0) & (yy < part["height"]) & (xx >= 0) & (xx < part["width"])
                    values[i] = np.where(inside, 1000 + yy * 100 + xx, 65000)
                if name == "source_valid":
                    values[i] = 1
                elif name == "SCL":
                    values[i] = 4
                    if invalid:
                        values[i, :, :max(1, (end-start)//2)] = 9
                elif name in ("QA_PIXEL", "QA_RADSAT"):
                    values[i] = 0
                    if invalid:
                        values[i, :, :max(1, (end-start)//2)] = 8
            with rasterio.open(
                path, "w", driver="GTiff", width=end-start, height=height,
                count=len(part["bands"]), dtype=part["dtype"], crs=part["crs"],
                transform=Affine(*part["transform"]) * Affine.translation(start, -top),
            ) as dst:
                dst.write(values)
                dst.descriptions = tuple(part["bands"])
            paths[part["id"]].append(path)
    return paths


@pytest.mark.parametrize("key", ["s1", "s2", "landsat"])
@pytest.mark.parametrize("border", [(0, 1, 0, 0), (1, 1, 1, 1)])
def test_native_rounded_border_preserves_exact_pixel_values(native_setup, key, border):
    cache, scene, job = native_setup(key)
    paths = write_parts(cache.root, job, split=True, coordinates=True)
    with NativeReader(job, paths) as reader:
        expected, mask = reader.read(job["transform"], 120, 120)
    paths = write_parts(cache.root, job, split=True, coordinates=True, border=border)
    with NativeReader(job, paths) as reader:
        actual, valid = reader.read(job["transform"], 120, 120)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(valid, mask)
    ingest_export(cache, job, paths)
    assert cache.done(scene).all()


@pytest.mark.parametrize("problem", ["large_border", "missing", "overlap"])
def test_border_tolerance_cannot_hide_bad_parts(native_setup, problem):
    cache, scene, job = native_setup("s1")
    paths = write_parts(
        cache.root, job, split=True, border=(0, 2 if problem == "large_border" else 1, 0, 0),
    )
    files = paths[job["parts"][0]["id"]]
    if problem == "missing":
        files.pop()
    elif problem == "overlap":
        files.append(files[0])
    with pytest.raises((ValueError, IncompleteExport)):
        ingest_export(cache, job, paths)
    assert not cache.done(scene).any()


@pytest.mark.parametrize("key,parts", [("s1", 1), ("s2", 3), ("landsat", 2)])
def test_native_reconstructs_across_file_seams(native_setup, key, parts):
    cache, scene, job = native_setup(key)
    paths = write_parts(cache.root, job, split=True)
    assert len(job["parts"]) == parts
    if key != "s1":
        assert job["raw_bytes"] < job["legacy_raw_bytes"] / 2
    ingest_export(cache, job, paths)
    group = cache._open(scene)
    expected = normalize_pixels(
        key, np.full((len(BANDS[key]), 120, 120), 100, "f4"),
        np.ones((120, 120), bool), cache.norm,
    )
    np.testing.assert_array_equal(group["values"][:], expected)
    assert group["valid"][:].all() and cache.done(scene).all()


@pytest.mark.parametrize("key", ["s2", "landsat"])
def test_local_qa_and_source_mask_exclude_invalid_but_keep_zero(native_setup, key):
    cache, _, job = native_setup(key)
    paths = write_parts(cache.root, job, invalid=True)
    p = job["parts"][0]
    with rasterio.open(paths[p["id"]][0], "r+") as dst:
        raw = dst.read()
        raw[0] = 0  # A valid measured zero must not become nodata.
        raw[-1, :3] = 0
        dst.write(raw)
    with NativeReader(job, paths) as reader:
        values, valid = reader.read(job["transform"], 120, 120)
    assert 0 < valid.sum() < valid.size
    assert not valid[0].any()
    assert (values[0][valid] == 0).all()


def test_missing_or_shifted_group_never_commits_cache(native_setup):
    cache, scene, job = native_setup("s2")
    paths = write_parts(cache.root, job)
    last = job["parts"][-1]["id"]
    missing = dict(paths, **{last: []})
    with pytest.raises(IncompleteExport):
        ingest_export(cache, job, missing)
    assert not cache.done(scene).any()
    with rasterio.open(paths[last][0], "r+") as dst:
        dst.transform *= Affine.translation(0.5, 0)
    with pytest.raises(ValueError, match="not aligned"):
        ingest_export(cache, job, paths)
    assert not cache.done(scene).any()


def test_exact_pixel_centres_select_source_cells_and_zero_is_valid(native_setup):
    cache, _, job = native_setup("landsat")
    paths = write_parts(cache.root, job)
    for part in job["parts"]:
        with rasterio.open(paths[part["id"]][0], "r+") as dst:
            raw = dst.read()
            for i, name in enumerate(part["bands"]):
                if name in BANDS["landsat"]:
                    raw[i] = np.indices(raw[i].shape)[1]
            dst.write(raw)
    with NativeReader(job, paths) as reader:
        values, valid = reader.read(job["transform"], 120, 120)
    assert valid.all()
    # Independently derive the source column from a destination centre.
    for i, band in enumerate(BANDS["landsat"]):
        part = next(p for p in job["parts"] if band in p["bands"])
        expected = [int(np.floor(
            (job["transform"][2] + (x+0.5)*10 - part["transform"][2]) / part["transform"][0]
        )) for x in range(120)]
        np.testing.assert_array_equal(values[i, 0], expected)


def test_native_queue_recovers_between_groups_without_duplicate_submission(
    native_setup, monkeypatch
):
    cache, scene, job = native_setup("s2")
    paths = write_parts(cache.root, job)
    settings = json.loads(Path("configs/data/gee_drive_native.json").read_text())
    settings["poll_seconds"] = 0.001
    monkeypatch.setattr("time.sleep", lambda _: None)
    submitted, cleaned = [], []
    fail_second = True
    state_path = cache.root / "exports.sqlite"
    state = ExportState(state_path)

    def submit(part, tid):
        submitted.append(part["id"])
        return tid

    def status(tid):
        nonlocal fail_second
        if len(submitted) == 2 and fail_second:
            fail_second = False
            raise RuntimeError("simulated restart")
        return {"state": "COMPLETED"}

    def cleanup(record):
        assert cache.done(scene).all()
        assert len(record["files"]) == 3
        cleaned.append(record["id"])

    def fetch(pid, files):
        assert all(f["export_id"] == pid for f in files)
        return paths[pid]

    transport = SimpleNamespace(
        ee=SimpleNamespace(data=SimpleNamespace(newTaskId=lambda: ["task"+str(len(submitted))])),
        ensure_folder=lambda: None, available_bytes=lambda: 1e12,
        submit=submit, status=status, cleanup=cleanup, fetch=fetch,
        inventory=lambda pid: [
            dict(id=pid, name=pid+".tif", size=str(paths[pid][0].stat().st_size), md5Checksum=pid)
        ],
    )
    # Inject the failure without retry swallowing it, as a real process interruption.
    monkeypatch.setattr("hydrostate.data.gee.workflows.drive_queue.retry", lambda fn: fn())
    with pytest.raises(RuntimeError, match="restart"):
        run_jobs(transport, state, cache, [job], settings)
    assert not cache.done(scene).any() and not cleaned
    assert state.get(job["id"])["part_index"] == 1
    state.close()
    state = ExportState(state_path)
    run_jobs(transport, state, cache, [], settings)
    assert submitted == [p["id"] for p in job["parts"]]
    assert cleaned == [job["id"]]
    assert not state.pending()
    state.close()


def test_layout_config_validation(native_setup):
    cache, _, _ = native_setup("s1")
    settings = json.loads(Path("configs/data/gee_drive_native.json").read_text())
    validate_settings(settings, cache)
    settings["export_layout"] = "native_typo"
    with pytest.raises(ValueError, match="export_layout"):
        validate_settings(settings, cache)


def test_native_mask_mismatch_cannot_publish_chunk(native_setup):
    from hydrostate.storage.quality_cache import QualityCache

    cache, scene, job = native_setup("s2")
    paths = write_parts(cache.root, job)
    quality = QualityCache(cache)
    quality.write_tile(scene, 0, 0, np.zeros((120, 120), "u1"))
    with pytest.raises(ValueError, match="differs from selection"):
        ingest_export(cache, job, paths)
    assert not cache.done(scene).any()


def test_native_budget_uses_transport_bytes_and_skips_reserved(native_setup):
    cache, scene, _ = native_setup("s2")
    tr = input_transform(cache.grid)
    groups = band_groups(scene, source_info("s2", (tr.c, tr.f)))
    demand = [dict(scene=scene, chunks={(0, 0)})]
    # Smaller than one aligned chunk, but enough for the actual native requests.
    jobs = list(selected_export_jobs(
        cache, demand, max_export_bytes=300000,
        job_transform=lambda job: native_job(job, groups),
    ))
    assert len(jobs) == 1
    assert jobs[0]["raw_bytes"] < 300000 < jobs[0]["legacy_raw_bytes"]
    assert list(selected_export_jobs(
        cache, demand, max_export_bytes=300000,
        job_transform=lambda job: native_job(job, groups),
        reserved_chunks={scene["asset_id"]: {(0, 0)}},
    )) == []


def test_legacy_receipt_schema_migrates_without_changing_payload(tmp_path):
    import sqlite3

    path = tmp_path / "old.sqlite"
    db = sqlite3.connect(path)
    db.execute("""CREATE TABLE exports (
        id TEXT PRIMARY KEY, payload TEXT NOT NULL, status TEXT NOT NULL,
        task_id TEXT, files TEXT, attempts INTEGER NOT NULL DEFAULT 0,
        updated TEXT NOT NULL, error TEXT)""")
    job = {"id": "legacy"}
    db.execute("INSERT INTO exports(id,payload,status,updated) VALUES(?,?,?,?)",
               ("legacy", json.dumps(job), "submitted", "old"))
    db.commit()
    db.close()
    state = ExportState(path)
    assert state.get("legacy")["payload"] == job
    assert state.get("legacy")["part_index"] == 0
    assert state.get("legacy")["status"] == "submitted"
    state.close()
