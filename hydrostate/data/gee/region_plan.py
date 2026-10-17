"""Bounded, chunk-aligned regional exports; no Earth Engine calls are made here."""

import math

import numpy as np

from hydrostate.data.gee.contracts import fingerprint, input_shape, input_transform, map_point
from hydrostate.data.gee.products import BANDS
from hydrostate.inference.planning import window_geometry


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
