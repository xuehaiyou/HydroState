"""Reuse aligned native pixels across regional plans without resampling or renormalizing."""

import math
import threading
from functools import lru_cache
from pathlib import Path

import numpy as np

from hydrostate.data.regional.plan import block_grid, load_plan, product
from hydrostate.data.retrieval.contract import INPUT_GROUPS
from hydrostate.storage.regional import product_path, ready
from hydrostate.storage.spatial_cache import SpatialChunkCache


class RegionalReuse:
    def __init__(self, plan, roots):
        self.plan = plan
        self.sources = [(str(Path(root).resolve()), load_plan(root)) for root in roots]
        self.local = threading.local()
        for root, source in self.sources:
            for key in ("version", "input_contract", "input_profile", "normalization", "groups"):
                if source[key] != plan[key]:
                    raise ValueError(f"Reuse source differs in {key}: {root}")
            a, b = source["grid"], plan["grid"]
            if a["crs"] != b["crs"] or any((a[k] - b[k]) % 60 for k in ("left", "top")):
                raise ValueError(f"Reuse requires the same CRS and aligned native grids: {root}")

    @lru_cache(maxsize=32768)
    def _ready(self, source_index, group, slot, block, orbit):
        root, plan = self.sources[source_index]
        item = product(plan, group, slot, block, orbit)
        return item if ready(root, item) else None

    def read(self, item):
        spec = INPUT_GROUPS[item["group"]]
        grid = block_grid(self.plan, item["block"], spec.resolution)
        height, width = grid["height"], grid["width"]
        covered, pieces = np.zeros((height, width), bool), []
        x, y = grid["transform"][2], grid["transform"][5]
        r = spec.resolution
        for source_index, (root, source) in enumerate(self.sources):
            sg, size = source["grid"], source["block_size"]
            x0, y0 = (x - sg["left"]) // 10, (sg["top"] - y) // 10
            x1, y1 = x0 + width * r // 10, y0 + height * r // 10
            for by in range(
                max(0, y0 // size), min(math.ceil(y1 / size), math.ceil(sg["height"] / size))
            ):
                for bx in range(
                    max(0, x0 // size), min(math.ceil(x1 / size), math.ceil(sg["width"] / size))
                ):
                    old = self._ready(
                        source_index,
                        item["group"],
                        tuple(item["slot"]),
                        (by, bx),
                        tuple(item["orbit"]) if item["orbit"] else None,
                    )
                    if old is None:
                        continue
                    old_grid = block_grid(source, (by, bx), r)
                    ox, oy = old_grid["transform"][2], old_grid["transform"][5]
                    dx, dy = (ox - x) // r, (y - oy) // r
                    a, b = max(0, dy), max(0, dx)
                    c, d = min(height, dy + old_grid["height"]), min(width, dx + old_grid["width"])
                    if c <= a or d <= b or covered[a:c, b:d].all():
                        continue
                    pieces.append((root, old, a, c, b, d, dy, dx))
                    covered[a:c, b:d] = True
        if not covered.all():
            # Download one whole missing block; do not issue fragmented edge RPCs.
            return None
        arrays = dict(
            values=np.zeros((len(spec.bands), height, width), "f2"),
            valid=np.zeros((1, height, width), "u1"),
            quality=np.zeros((4, height, width), "f2"),
        )
        if not hasattr(self.local, "cache"):
            self.local.cache = SpatialChunkCache(max_bytes=256 * 1024**2)
        cache, provenance = self.local.cache, []
        covered[:] = False
        for root, old, a, c, b, d, dy, dx in pieces:
            remaining = ~covered[a:c, b:d]
            for name in arrays:
                value = cache.read(
                    product_path(root, old), name, slice(a - dy, c - dy), slice(b - dx, d - dx)
                )
                arrays[name][:, a:c, b:d][:, remaining] = value[:, remaining]
            provenance.append(
                dict(
                    root=root,
                    product=old["id"],
                    destination=[a, c, b, d],
                    pixels=int(remaining.sum()),
                )
            )
            covered[a:c, b:d] = True
        return arrays, provenance
