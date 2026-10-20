"""Bounded, chunk-aligned regional exports; no Earth Engine calls are made here."""

import json
import math

import numpy as np

from hydrostate.data.gee.contracts import fingerprint, input_shape, input_transform, map_point
from hydrostate.data.gee.products import BANDS
from hydrostate.inference.planning import window_geometry


def selected_chunks(cache, rows):
    """Union the selected scenes' input chunks, including each window's halo."""
    demands = {}
    h, w = input_shape(cache.grid)
    for row in rows:
        if not row["accepted"]:
            continue
        y, x = int(row["row"]) * 3, int(row["col"]) * 3
        if y < 0 or x < 0 or y + 120 > h or x + 120 > w:
            raise ValueError("Selected window outside input grid")
        cells = {
            (cy, cx)
            for cy in range(y // cache.chunk, (y + 119) // cache.chunk + 1)
            for cx in range(x // cache.chunk, (x + 119) // cache.chunk + 1)
        }
        for selected in json.loads(row["selections"]).values():
            if selected is None:
                continue
            scene = {k: v for k, v in selected.items() if k != "valid_fraction"}
            asset = scene["asset_id"]
            entry = demands.setdefault(asset, dict(scene=scene, chunks=set()))
            if entry["scene"] != scene:
                raise ValueError(f"Conflicting metadata for {asset}")
            entry["chunks"].update(cells)
    return [demands[key] for key in sorted(demands)]


def selected_export_jobs(
    cache, demands, *, max_export_bytes, max_padding_fraction=0.25, reserved_chunks=None,
    job_transform=None, export_scope="selected_chunks",
):
    """Split sparse demand into rectangles bounded by bytes and extra pixel area.

    The budget and padding apply to actual pixels, including partial edge chunks.
    Completed chunks and chunks reserved by recoverable in-flight exports are
    never demanded. This also deduplicates overlapping dates before completion.
    The low-level default preserves the previous policy; workflows pass their config.
    """
    if not math.isfinite(max_export_bytes) or max_export_bytes <= 0:
        raise ValueError("Invalid export byte budget")
    if not math.isfinite(max_padding_fraction) or max_padding_fraction < 0:
        raise ValueError("Invalid padding fraction")
    if export_scope not in ("selected_chunks", "scene_region"):
        raise ValueError("Invalid export_scope")
    h, w = input_shape(cache.grid)
    chunk, tr = cache.chunk, input_transform(cache.grid)
    reserved_chunks = reserved_chunks or {}
    for demand in demands:
        scene = demand["scene"]
        done = cache.done(scene)
        reserved = reserved_chunks.get(scene["asset_id"], set())
        needed = sorted(c for c in demand["chunks"] if not done[c] and c not in reserved)
        if not needed:
            continue
        if export_scope == "scene_region":
            # The catalog bounds are already in the regional projected CRS.
            # Include selected window halos even where they cross the scene edge.
            cells = scene_region_chunks(cache, scene) | set(needed)
            needed = sorted(c for c in cells if not done[c] and c not in reserved)
        bands = [*BANDS[scene["key"]], "valid"]

        def pack(cells):
            if not cells:
                return
            cy0, cx0 = min(c[0] for c in cells), min(c[1] for c in cells)
            cy1, cx1 = max(c[0] for c in cells) + 1, max(c[1] for c in cells) + 1
            y, x = cy0 * chunk, cx0 * chunk
            height, width = min(h, cy1 * chunk) - y, min(w, cx1 * chunk) - x
            area = width * height
            useful = sum(
                min(chunk, h - cy * chunk) * min(chunk, w - cx * chunk) for cy, cx in cells
            )
            raw_bytes = area * len(bands) * 4
            if export_scope == "scene_region" or area <= useful * (1 + max_padding_fraction):
                left, top = map_point(tr, x, y)
                job = dict(
                    scene=scene,
                    contract=cache.contract,
                    grid=cache.grid,
                    chunk=chunk,
                    bands=bands,
                    chunks=sorted(map(list, cells)),
                    x=x,
                    y=y,
                    width=width,
                    height=height,
                    transform=[10, 0, left, 0, -10, top],
                    raw_bytes=raw_bytes,
                    preprocessing="gee_sampling.prepare_v1",
                    version=2,
                )
                if export_scope == "scene_region":
                    job.update(export_scope=export_scope, version=3)
                job["id"] = fingerprint(job)[:24]
                if job_transform is not None:
                    job = job_transform(job)
                if job["raw_bytes"] <= max_export_bytes:
                    yield job
                    return
            if len(cells) == 1:
                raise ValueError("Export byte budget is smaller than one requested chunk")
            axis = 0 if cy1 - cy0 >= cx1 - cx0 else 1
            midpoint = (cy0 + cy1) // 2 if axis == 0 else (cx0 + cx1) // 2
            yield from pack([c for c in cells if c[axis] < midpoint])
            yield from pack([c for c in cells if c[axis] >= midpoint])

        yield from pack(needed)


def scene_region_chunks(cache, scene):
    """Chunk-aligned intersection of the source bounds and regional input grid.

    Cache the entire requested rectangle so later dates can reuse pixels that
    were not selected by the first date. Source masks retain the true footprint.
    """
    left, bottom, right, top = scene["bounds"]
    if not np.isfinite([left, bottom, right, top]).all() or left >= right or bottom >= top:
        raise ValueError("Invalid projected scene bounds")
    h, w = input_shape(cache.grid)
    tr = input_transform(cache.grid)
    cy0 = max(0, math.floor((tr.f - top) / (10 * cache.chunk)))
    cx0 = max(0, math.floor((left - tr.c) / (10 * cache.chunk)))
    cy1 = min(math.ceil(h / cache.chunk), math.ceil((tr.f - bottom) / (10 * cache.chunk)))
    cx1 = min(math.ceil(w / cache.chunk), math.ceil((right - tr.c) / (10 * cache.chunk)))
    return {(cy, cx) for cy in range(cy0, cy1) for cx in range(cx0, cx1)}


def export_jobs(cache, scenes, windows, *, max_export_bytes):
    """Pack missing scene chunks into spatial rectangles with a bounded raw size."""
    chunk = cache.chunk
    h, w = input_shape(cache.grid)
    tr = input_transform(cache.grid)
    footprints = np.array([window_geometry(cache.grid, window).bounds for window in windows])
    if len(footprints) == 0:
        return
    for scene in scenes:
        done = cache.done(scene)
        needed = set()
        left, bottom, right, top = scene["bounds"]
        intersects = (
            (footprints[:, 0] <= right)
            & (footprints[:, 2] >= left)
            & (footprints[:, 1] <= top)
            & (footprints[:, 3] >= bottom)
        )
        for index in np.flatnonzero(intersects):
            window = windows[index]
            y, x = window["row"] * 3, window["col"] * 3
            for cy in range(y // chunk, (y + 119) // chunk + 1):
                for cx in range(x // chunk, (x + 119) // chunk + 1):
                    if not done[cy, cx]:
                        needed.add((cy, cx))
        if not needed:
            continue
        bands = [*BANDS[scene["key"]], "valid"]
        bytes_per_chunk = chunk * chunk * len(bands) * 4
        if max_export_bytes < bytes_per_chunk:
            raise ValueError("Export byte budget is smaller than one cache chunk")
        capacity = int(max_export_bytes // bytes_per_chunk)
        # Group a fixed spatial grid: restarts and dates preserve rectangle alignment.
        columns = min(math.ceil(w / chunk), capacity)
        rows = max(1, capacity // columns)
        groups = {}
        for cy, cx in needed:
            groups.setdefault((cy // rows, cx // columns), []).append((cy, cx))
        for cells in groups.values():
            y0, x0 = min(c[0] for c in cells) * chunk, min(c[1] for c in cells) * chunk
            y1 = min(h, (max(c[0] for c in cells) + 1) * chunk)
            x1 = min(w, (max(c[1] for c in cells) + 1) * chunk)
            left, top = map_point(tr, x0, y0)
            job = dict(
                scene=scene,
                contract=cache.contract,
                grid=cache.grid,
                chunk=chunk,
                bands=bands,
                chunks=sorted(map(list, cells)),
                x=x0,
                y=y0,
                width=x1 - x0,
                height=y1 - y0,
                transform=[10, 0, left, 0, -10, top],
                raw_bytes=(x1 - x0) * (y1 - y0) * len(bands) * 4,
                preprocessing="gee_sampling.prepare_v1",
                version=1,
            )
            job["id"] = fingerprint(job)[:24]
            yield job
