"""Whole scene-region requests preserve cache reuse and bounded fallback."""

from types import SimpleNamespace

import numpy as np
import pytest

from hydrostate.data.gee.contracts import input_transform
from hydrostate.data.gee.native import native_job
from hydrostate.data.gee.region_plan import selected_export_jobs


def setup():
    grid = dict(crs="EPSG:32650", left=400000, top=3200000, height=312, width=312, halo=4)
    done = np.zeros((2, 2), bool)
    cache = SimpleNamespace(grid=grid, chunk=480, contract="test", done=lambda _: done)
    tr = input_transform(grid)
    scene = dict(
        key="s1",
        asset_id="test",
        time=1,
        bounds=[tr.c + 1, tr.f - 9599, tr.c + 9599, tr.f - 1],
    )
    groups = [
        dict(crs=grid["crs"], source_transform=list(tr)[:6], bands=["VV", "VH"], dtype="float32")
    ]
    return cache, scene, done, lambda j: native_job(j, groups)


def test_scene_region_prefetches_coverage_for_future_dates():
    cache, scene, done, convert = setup()
    kwargs = dict(max_export_bytes=1e8, job_transform=convert, export_scope="scene_region")
    jobs = list(selected_export_jobs(cache, [dict(scene=scene, chunks={(0, 0)})], **kwargs))
    assert len(jobs) == 1
    job = jobs[0]
    assert job["chunks"] == [[0, 0], [0, 1], [1, 0], [1, 1]]
    assert job["width"] == job["height"] == 960
    # Later dates use reservations; no duplicate task while the first is running.
    assert not list(
        selected_export_jobs(
            cache,
            [dict(scene=scene, chunks={(1, 1)})],
            reserved_chunks={"test": set(map(tuple, job["chunks"]))},
            **kwargs,
        )
    )
    done[:] = True
    assert not list(selected_export_jobs(cache, [dict(scene=scene, chunks={(1, 1)})], **kwargs))


def test_scene_region_splits_by_native_bytes_and_preserves_legacy_reservations():
    cache, scene, done, convert = setup()
    done[0, 0] = True
    demand = [dict(scene=scene, chunks={(1, 1)})]
    cap = 482 * 482 * 3 * 4
    jobs = list(
        selected_export_jobs(
            cache,
            demand,
            max_export_bytes=cap,
            job_transform=convert,
            export_scope="scene_region",
            reserved_chunks={"test": {(0, 1)}},
        )
    )
    assert len(jobs) == 2
    assert {tuple(c) for j in jobs for c in j["chunks"]} == {(1, 0), (1, 1)}
    assert all(j["raw_bytes"] <= cap for j in jobs)
    assert not list(
        selected_export_jobs(
            cache,
            [dict(scene=scene, chunks={(0, 0)})],
            max_export_bytes=cap,
            job_transform=convert,
            export_scope="scene_region",
        )
    ), "Do not prefetch unselected scenes when the date's demand is already cached"


def test_scene_extent_is_clipped_but_selected_halo_is_not_lost():
    cache, scene, _, convert = setup()
    tr = input_transform(cache.grid)
    scene["bounds"] = [tr.c - 10000, tr.f - 4700, tr.c + 4700, tr.f + 10000]
    jobs = list(
        selected_export_jobs(
            cache,
            [dict(scene=scene, chunks={(1, 0)})],
            max_export_bytes=1e8,
            job_transform=convert,
            export_scope="scene_region",
        )
    )
    assert len(jobs) == 1
    assert jobs[0]["chunks"] == [[0, 0], [1, 0]]
    assert jobs[0]["width"] == 480 and jobs[0]["height"] == 960
    with pytest.raises(ValueError, match="export_scope"):
        list(selected_export_jobs(cache, [], max_export_bytes=1e8, export_scope="typo"))
