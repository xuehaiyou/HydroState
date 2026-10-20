"""Mask-first selection must preserve observations and never publish partial inputs."""

import json
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Lock
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from hydrostate.data.gee.backends import pixels
from hydrostate.data.gee.contracts import input_shape, input_transform
from hydrostate.data.gee.products import BANDS
from hydrostate.data.gee.region_plan import selected_chunks, selected_export_jobs
from hydrostate.data.gee.selection import epoch, select_window
from hydrostate.storage.quality_cache import QualityCache
from hydrostate.storage.scene_cache import SceneCache


@pytest.fixture
def setup(tmp_path):
    plan = dict(
        grid=dict(
            crs="EPSG:32650", left=400000, top=3200000, height=312, width=312, halo=4, core=32
        ),
        config=dict(
            cache_chunk=480,
            min_valid_fraction=0.5,
            min_active_slots=2,
            latest_days=3,
            gee_project="test",
        ),
        training=[],
    )
    scene = dict(
        key="s1",
        asset_id="test/s1",
        time=epoch("2025-01-21"),
        orbit=12,
        bounds=[0, 0, 10000000, 10000000],
    )
    return SceneCache(tmp_path, plan), scene


def write_full(cache, scene, cy, cx, valid):
    cache.write_chunk(
        scene, cy, cx, np.zeros((len(BANDS[scene["key"]]), *valid.shape), "f4"), valid
    )


def fake_pixels(monkeypatch, cache, fetch):
    monkeypatch.setattr(pixels, "validity_expression", lambda ee, scene: scene)
    origin = input_transform(cache.grid)

    def compute(ee, scene, names, crs, transform, width, height):
        assert names == ["valid"] and crs == cache.grid["crs"]
        y, x = round((origin.f - transform[5]) / 10), round((transform[2] - origin.c) / 10)
        result = np.empty((height, width), dtype=[("valid", "u1")])
        result["valid"] = fetch(scene, y, x, height, width)
        return result

    monkeypatch.setattr(pixels, "compute_pixels", compute)


def test_masks_preserve_boundary_pixels_and_never_complete_inputs(setup, monkeypatch):
    cache, scene = setup
    pattern = np.indices(input_shape(cache.grid)).sum(axis=0) % 3 == 1
    fake_pixels(monkeypatch, cache, lambda s, y, x, h, w: pattern[y : y + h, x : x + w])
    quality = pixels.PixelQualityCache(cache, ee=object(), tile_chunks=1)
    actual = quality.read_valid(scene, dict(row=156, col=156))
    np.testing.assert_array_equal(actual, pattern[468:588, 468:588])
    assert quality.requests == 4
    assert quality.response_bytes == 960**2
    assert not cache.path(scene).exists()
    assert not cache.done(scene).any()
    # Durable masks survive a new process/cache instance without any request.
    reopened = QualityCache(cache, tile_chunks=1)
    np.testing.assert_array_equal(reopened.read_valid(scene, dict(row=156, col=156)), actual)


def test_complete_full_chunks_bypass_mask_download(setup, monkeypatch):
    cache, scene = setup
    valid = np.indices((480, 480)).sum(axis=0) % 2 == 0
    write_full(cache, scene, 0, 0, valid)
    fake_pixels(monkeypatch, cache, lambda *a: pytest.fail("Already cached full inputs"))
    quality = pixels.PixelQualityCache(cache, ee=object())
    np.testing.assert_array_equal(quality.read_valid(scene, dict(row=0, col=0)), valid[:120, :120])
    assert not quality.path(scene).exists()


def test_requests_deduplicate_but_different_tiles_run_concurrently(setup, monkeypatch):
    cache, scene = setup
    barrier, guard, calls = Barrier(2, timeout=10), Lock(), []

    def fetch(s, y, x, h, w):
        with guard:
            calls.append((y, x))
        barrier.wait()  # Deadlocks/fails if a cache/write lock encloses network I/O.
        return np.ones((h, w), "u1")

    fake_pixels(monkeypatch, cache, fetch)
    quality = pixels.PixelQualityCache(cache, ee=object(), tile_chunks=1)
    with ThreadPoolExecutor(4) as pool:
        jobs = [pool.submit(quality.get_tile, scene, 0, x) for x in [0, 0, 0, 1]]
        assert all(f.result().all() for f in jobs)
    assert sorted(calls) == [(0, 0), (0, 480)]
    assert quality.requests == 2


def test_selection_matches_full_masks_and_keeps_orbit_and_recency(setup, monkeypatch):
    cache, scene = setup
    older = dict(scene, asset_id="older", time=epoch("2025-01-02"))
    wrong_orbit = dict(older, asset_id="wrong", time=epoch("2025-01-03"), orbit=13)
    too_old = dict(older, asset_id="unused", time=epoch("2025-01-01"))
    cloudy = dict(scene, key="s2", asset_id="cloudy")
    clear = dict(cloudy, asset_id="clear", time=epoch("2025-01-20"))
    scenes = dict(s1=[scene, wrong_orbit, older, too_old], s2=[cloudy, clear], landsat=[])
    pattern = np.indices((960, 960)).sum(axis=0) % 7 != 0
    masks = {s["asset_id"]: pattern for group in scenes.values() for s in group}
    masks["cloudy"] = np.zeros((960, 960), bool)
    requested = []

    def fetch(s, y, x, h, w):
        requested.append(s["asset_id"])
        return masks[s["asset_id"]][y : y + h, x : x + w]

    fake_pixels(monkeypatch, cache, fetch)
    quality = pixels.PixelQualityCache(cache, ee=object())
    full = SimpleNamespace(read_valid=lambda s, row: masks[s["asset_id"]][468:588, 468:588])
    window = dict(window_id="w", row=156, col=156)
    period = dict(time="2025-01-21", start="2025-01-17", end="2025-01-25")
    expected = select_window(cache.plan, period, window, scenes, full)
    assert expected["accepted"]
    assert select_window(cache.plan, period, window, scenes, quality) == expected
    assert "wrong" not in requested and "unused" not in requested
    assert "cloudy" in requested and "clear" in requested


def test_selected_demand_skips_rejected_windows_and_includes_halo(setup):
    cache, scene = setup
    other = dict(scene, asset_id="rejected")
    row = dict(
        row=156,
        col=156,
        accepted=True,
        selections=json.dumps(dict(s1_0=dict(scene, valid_fraction=0.75))),
    )
    rejected = dict(row, accepted=False, selections=json.dumps(dict(s1_0=other)))
    demands = selected_chunks(cache, [row, row, rejected])
    assert len(demands) == 1
    assert demands[0]["scene"] == scene
    assert demands[0]["chunks"] == {(0, 0), (0, 1), (1, 0), (1, 1)}
    write_full(cache, scene, 0, 0, np.ones((480, 480), bool))
    jobs = list(selected_export_jobs(cache, demands, max_export_bytes=480**2 * 12))
    assert len(jobs) == 3
    assert all(j["raw_bytes"] <= 480**2 * 12 for j in jobs)
    assert all([0, 0] not in j["chunks"] for j in jobs)
    changed_fraction = dict(row, selections=json.dumps(dict(s1_0=dict(scene, valid_fraction=1))))
    assert (
        list(
            selected_export_jobs(
                cache, selected_chunks(cache, [changed_fraction]), max_export_bytes=480**2 * 12
            )
        )
        == jobs
    )


def test_sparse_exports_limit_padding_and_cover_partial_edges(tmp_path):
    plan = dict(
        grid=dict(crs="EPSG:32650", left=0, top=0, height=1000, width=1000, halo=4),
        config=dict(cache_chunk=480),
    )
    cache = SceneCache(tmp_path, plan)
    scene = dict(key="landsat", asset_id="sparse")
    cells = {(0, 0), (0, 1), (1, 0), (6, 6)}
    demands = [dict(scene=scene, chunks=cells)]
    jobs = list(selected_export_jobs(cache, demands, max_export_bytes=1e9))
    assert {tuple(c) for j in jobs for c in j["chunks"]} == cells
    assert sum(len(j["chunks"]) for j in jobs) == len(cells)
    assert len(jobs) >= 3
    for job in jobs:
        useful = sum(
            min(480, 3024 - cy * 480) * min(480, 3024 - cx * 480) for cy, cx in job["chunks"]
        )
        assert job["width"] * job["height"] <= useful * 1.25
    edge = next(j for j in jobs if [6, 6] in j["chunks"])
    assert edge["width"] == edge["height"] == 144


def test_verify_requires_full_values_and_detects_mask_mismatch(setup):
    cache, scene = setup
    quality = QualityCache(cache, tile_chunks=1)
    quality.write_tile(scene, 0, 0, np.ones((480, 480), bool))
    demand = [dict(scene=scene, chunks={(0, 0)})]
    with pytest.raises(FileNotFoundError, match="Selected input"):
        quality.verify_full(demand)
    write_full(cache, scene, 0, 0, np.zeros((480, 480), bool))
    with pytest.raises(ValueError, match="mask mismatch"):
        quality.verify_full(demand)


def test_drive_workflow_resumes_selection_and_publishes_only_complete_inputs(setup, monkeypatch):
    from hydrostate.data.gee import client
    from hydrostate.data.gee.backends import drive
    from hydrostate.data.gee.workflows import drive_queue
    from hydrostate.data.regional import common
    from hydrostate.data.regional import gee_drive as region

    cache, scene = setup
    older = dict(scene, asset_id="older", time=epoch("2025-01-02"))
    period = dict(time="2025-01-21", start="2025-01-17", end="2025-01-25")
    monkeypatch.setattr(client, "initialize", lambda _: object())
    monkeypatch.setattr(drive, "DriveTransport", lambda *a, **k: object())
    monkeypatch.setattr(
        region,
        "windows",
        lambda _: [
            dict(window_id=f"{r}_{c}", row=r, col=c) for r, c in [(0, 0), (0, 160), (160, 0)]
        ],
    )
    monkeypatch.setattr(
        region,
        "catalog",
        lambda root, plan, key, year, ee: [scene, older] if key == "s1" and year == 2025 else [],
    )
    monkeypatch.setattr(common, "windows", region.windows)
    fake_pixels(monkeypatch, cache, lambda s, y, x, h, w: np.ones((h, w), bool))
    settings = dict(
        max_exports=2,
        download_workers=2,
        file_dimensions=3840,
        max_export_gib=0.5,
        max_padding_fraction=0.5,
        max_pending_gib=1,
        drive_reserve_gib=0.5,
        poll_seconds=1,
    )
    fail = True

    def run(transport, state, full, jobs, settings, *, on_progress):
        jobs = list(jobs)
        assert (cache.root / "selections/2025-01-21.parquet").exists()
        assert not (cache.root / "manifests/2025-01-21.parquet").exists()
        # A three-chunk L shape merges into one rectangle per scene only with 50% padding.
        assert len(jobs) == 2
        assert all(len(j["chunks"]) == 3 and j["width"] == j["height"] == 960 for j in jobs)
        if fail:
            raise OSError("interrupted export")
        for job in jobs:
            for cy, cx in job["chunks"]:
                write_full(full, job["scene"], cy, cx, np.ones((480, 480), bool))
        on_progress()

    monkeypatch.setattr(drive_queue, "run_jobs", run)
    with pytest.raises(OSError, match="interrupted"):
        region.prepare_region_drive(cache.root, cache.plan, [period], settings)
    assert not (cache.root / "manifests/2025-01-21.parquet").exists()
    report = json.loads((cache.root / "reports/preparation_2025-01-21.json").read_text())
    assert report["status"] == "selected" and report["mask_requests"] == 2
    fail = False
    monkeypatch.setattr(common, "select_window", lambda *a: pytest.fail("Selection already saved"))
    region.prepare_region_drive(cache.root, cache.plan, [period], settings)
    assert pd.read_parquet(cache.root / "manifests/2025-01-21.parquet")["accepted"].all()
    report = json.loads((cache.root / "reports/preparation_2025-01-21.json").read_text())
    assert report["status"] == "ready" and report["mask_requests"] == 0
    assert report["mask_verified_chunks"] == 6
    assert report["packing"] == dict(max_export_gib=0.5, max_padding_fraction=0.5)
    for s in (scene, older):
        assert not cache.done(s)[1, 1]


@pytest.mark.parametrize("recover", [False, True])
@pytest.mark.parametrize("lookahead", [1, 2, 4])
def test_drive_dates_fill_idle_slot_deduplicate_and_bound_lookahead(
    setup, monkeypatch, recover, lookahead
):
    from hydrostate.data.gee import client
    from hydrostate.data.gee.backends import drive
    from hydrostate.data.gee.state import ExportState
    from hydrostate.data.gee.workflows import drive_queue
    from hydrostate.data.regional import gee_drive as region

    cache, scene = setup
    scenes = [dict(scene, asset_id=f"scene-{key}") for key in "abc"]
    periods = [dict(time=when) for when in ("2025-01-21", "2025-01-29", "2025-02-06")]
    # The second date shares the slow first scene and adds its own independent scene.
    selected = [[scenes[0]], scenes[:2], [scenes[2]]]
    tables = {
        period["time"]: pd.DataFrame([dict(
            window_id="0_0", row=0, col=0, accepted=True,
            selections=json.dumps({str(i): s for i, s in enumerate(group)}),
        )])
        for period, group in zip(periods, selected, strict=True)
    }
    planned, submitted, cleaned, polls = [], [], [], {}
    settings = dict(
        max_exports=2, max_pending_dates=lookahead, download_workers=2,
        file_dimensions=3840, max_export_gib=0.5, max_pending_gib=1,
        drive_reserve_gib=0.5, poll_seconds=0.001,
    )
    monkeypatch.setattr(client, "initialize", lambda _: object())
    monkeypatch.setattr(region, "windows", lambda _: [dict(window_id="0_0", row=0, col=0)])
    monkeypatch.setattr(region, "catalog", lambda *a: [])
    monkeypatch.setattr(drive_queue.time, "sleep", lambda _: None)
    fake_pixels(monkeypatch, cache, lambda *a: pytest.fail("Saved selections need no masks"))

    def prepare(root, plan, period, *args, **kwargs):
        when = period["time"]
        unpublished = [
            d for d in planned if not (root / "manifests" / f"{d}.parquet").exists()
        ]
        assert len(unpublished) < lookahead
        planned.append(when)
        region._publish_table(tables[when], root / "selections" / f"{when}.parquet")
        return tables[when]

    monkeypatch.setattr(region, "_prepare_windows", prepare)
    requests = {}
    if recover:
        job = next(selected_export_jobs(
            cache, [dict(scene=scenes[0], chunks={(0, 0)})], max_export_bytes=1e8
        ))
        state = ExportState(cache.root / "exports.sqlite")
        state.add(job)
        state.update(job["id"], status="submitted", task_id="retained-task")
        state.close()
        requests["retained-task"] = job

    def submit(job, task_id):
        asset = job["scene"]["asset_id"]
        assert asset not in submitted
        if recover:
            assert asset != "scene-a", "Existing cloud export must not be resubmitted"
        submitted.append(asset)
        requests[task_id] = job
        return task_id

    def status(task_id):
        asset = requests[task_id]["scene"]["asset_id"]
        polls[asset] = polls.get(asset, 0) + 1
        assert polls[asset] < 20, "Scheduler stalled instead of using the idle export slot"
        if asset == "scene-a":
            if lookahead == 1:
                assert planned == [periods[0]["time"]]
                complete = polls[asset] >= 3
            else:
                complete = "scene-b" in cleaned and polls[asset] >= 4
                if "scene-b" in cleaned and lookahead == 2:
                    assert planned == [p["time"] for p in periods[:2]]
                if "scene-c" in cleaned:
                    assert (cache.root / "manifests/2025-02-06.parquet").exists()
                    assert not (cache.root / "manifests/2025-01-21.parquet").exists()
            return dict(state="COMPLETED" if complete else "RUNNING")
        return dict(state="COMPLETED")

    def ingest(full, job, paths):
        asset = job["scene"]["asset_id"]
        if asset == "scene-b" and lookahead > 1:
            assert "scene-a" not in cleaned
            assert not list((cache.root / "manifests").glob("*.parquet"))
        for cy, cx in job["chunks"]:
            write_full(full, job["scene"], cy, cx, np.ones((480, 480), bool))

    transport = SimpleNamespace(
        ensure_folder=lambda: None, available_bytes=lambda: 10 * 1024**3,
        ee=SimpleNamespace(data=SimpleNamespace(
            newTaskId=lambda: [f"task-{len(submitted)}"],
        )),
        submit=submit, status=status,
        inventory=lambda jid: [dict(id=jid, size="1")],
        fetch=lambda *a: [],
        cleanup=lambda record: cleaned.append(record["payload"]["scene"]["asset_id"]),
    )
    monkeypatch.setattr(drive, "DriveTransport", lambda *a, **k: transport)
    monkeypatch.setattr(drive_queue, "ingest_export", ingest)
    region.prepare_region_drive(cache.root, cache.plan, periods, settings, min_free_gib=5)
    assert sorted(submitted) == (["scene-b", "scene-c"] if recover else [
        "scene-a", "scene-b", "scene-c"
    ])
    assert cleaned == {
        1: ["scene-a", "scene-b", "scene-c"],
        2: ["scene-b", "scene-a", "scene-c"],
        4: ["scene-b", "scene-c", "scene-a"],
    }[lookahead]
    for period in periods:
        report = json.loads((
            cache.root / "reports" / f"preparation_{period['time']}.json"
        ).read_text())
        assert report["status"] == "ready"
        assert (cache.root / "manifests" / f"{period['time']}.parquet").exists()
    report = json.loads((cache.root / "reports/preparation_2025-01-29.json").read_text())
    assert report["reserved_chunks"] == (1 if lookahead > 1 else 0)
    assert report["new_exports"] == 1
    state = ExportState(cache.root / "exports.sqlite")
    assert not state.pending()
    state.close()
