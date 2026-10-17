"""Offline, byte-mask Zarr cache for selection before spectral downloads."""

import math
import shutil
from collections import OrderedDict
from pathlib import Path
from threading import Event, RLock

import numpy as np
import zarr
from numcodecs import Blosc

from hydrostate.data.gee.contracts import fingerprint, input_shape, locked


class QualityCache:
    """Keep quality completion separate from full-input completion.

    Masks use the same 10 m grid as the full inputs. Larger tiles amortize small
    network requests; bounded in-memory caches avoid rereading Zarr per window.
    Full-band chunks already on disk take precedence and require no new masks.
    """

    def __init__(self, full_cache, *, root=None, tile_chunks=4):
        if not isinstance(tile_chunks, int) or not 1 <= tile_chunks <= 4:
            raise ValueError("tile_chunks must be an integer in [1,4]")
        self.full = full_cache
        self.root = Path(root) if root is not None else full_cache.root
        self.root.mkdir(parents=True, exist_ok=True)
        self.grid, self.chunk = full_cache.grid, full_cache.chunk
        self.tile = self.chunk * tile_chunks
        self.shape = input_shape(self.grid)
        self.contract = fingerprint(dict(full=full_cache.contract, tile=self.tile, version=1))
        self.stopped = Event()
        self.guard = RLock()
        self._tiles, self._chunks = OrderedDict(), OrderedDict()
        self._done = {}

    def path(self, scene):
        return (
            self.root / "quality_cache" / scene["key"] / (fingerprint(scene["asset_id"]) + ".zarr")
        )

    def refresh_full(self):
        """Call between selection stages, after full-band writes have completed."""
        with self.guard:
            self._done.clear()
            self._chunks.clear()

    def tile_bounds(self, ty, tx):
        h, w = self.shape
        y, x = ty * self.tile, tx * self.tile
        if not (0 <= y < h and 0 <= x < w):
            raise ValueError("Quality tile outside input grid")
        return y, x, min(h, y + self.tile), min(w, x + self.tile)

    @staticmethod
    def _remember(cache, key, value, capacity):
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > capacity:
            cache.popitem(last=False)
        return value

    def _open(self, scene, writable=False):
        group = zarr.open_group(str(self.path(scene)), mode="a" if writable else "r")
        if writable and "contract" not in group.attrs:
            compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
            group.require_dataset(
                "valid",
                shape=self.shape,
                chunks=(self.tile, self.tile),
                dtype="u1",
                compressor=compressor,
            )
            shape = tuple(math.ceil(n / self.tile) for n in self.shape)
            group.require_dataset(
                "done", shape=shape, chunks=shape, dtype="u1", compressor=compressor
            )
            group.attrs.update(contract=self.contract, scene=scene)
        if group.attrs.get("contract") != self.contract:
            raise ValueError("Quality cache grid/preprocessing mismatch")
        if group["valid"].chunks != (self.tile, self.tile):
            raise ValueError("Quality cache chunk layout mismatch")
        return group

    def get_tile(self, scene, ty, tx):
        """Read a completed mask tile; the pixel backend implements cache misses."""
        key = (scene["asset_id"], ty, tx)
        with self.guard:
            if key in self._tiles:
                self._tiles.move_to_end(key)
                return self._tiles[key]
            if not self.path(scene).exists():
                raise FileNotFoundError("Quality tile not prepared")
            group = self._open(scene)
            if not group["done"][ty, tx]:
                raise FileNotFoundError("Quality tile not prepared")
            y, x, y1, x1 = self.tile_bounds(ty, tx)
            valid = group["valid"][y:y1, x:x1].astype(bool)
            return self._remember(self._tiles, key, valid, 8)

    def write_tile(self, scene, ty, tx, valid):
        y, x, y1, x1 = self.tile_bounds(ty, tx)
        if valid.shape != (y1 - y, x1 - x) or not np.isin(valid, [0, 1]).all():
            raise ValueError("Invalid quality mask shape/values")
        if shutil.disk_usage(self.root).free < self.full.min_free_gib * 1024**3:
            raise OSError(f"Less than {self.full.min_free_gib:g} GiB free")
        # Network requests must finish before acquiring this write lock.
        with self.guard, locked(str(self.path(scene)) + ".lock"):
            group = self._open(scene, writable=True)
            if not group["done"][ty, tx]:
                group["valid"][y:y1, x:x1] = valid
                if not np.array_equal(group["valid"][y:y1, x:x1], valid):
                    raise OSError("Quality cache readback failed")
                self.full._sync_chunk(group["valid"], (ty, tx))
                group["done"][ty, tx] = 1
                self.full._sync_chunk(group["done"], (0, 0))
            else:
                valid = group["valid"][y:y1, x:x1]
            return self._remember(self._tiles, (scene["asset_id"], ty, tx), valid.astype(bool), 8)

    def _read_chunk(self, scene, cy, cx):
        asset = scene["asset_id"]
        key = (asset, cy, cx)
        y, x = cy * self.chunk, cx * self.chunk
        y1, x1 = min(self.shape[0], y + self.chunk), min(self.shape[1], x + self.chunk)
        with self.guard:
            if key in self._chunks:
                self._chunks.move_to_end(key)
                return self._chunks[key]
            if asset not in self._done:
                self._done[asset] = self.full.done(scene)
            if self._done[asset][cy, cx]:
                valid = self.full._open(scene)["valid"][y:y1, x:x1].astype(bool)
                return self._remember(self._chunks, key, valid, 128)
        ty, tx = y // self.tile, x // self.tile
        tile = self.get_tile(scene, ty, tx)
        # Copy so a cached small chunk cannot retain an evicted large tile.
        valid = tile[
            y % self.tile : y % self.tile + y1 - y, x % self.tile : x % self.tile + x1 - x
        ].copy()
        with self.guard:
            return self._remember(self._chunks, key, valid, 128)

    def read_valid(self, scene, window):
        if self.stopped.is_set():
            raise RuntimeError("Input preparation stopped after a worker failure")
        y, x = int(window["row"]) * 3, int(window["col"]) * 3
        if y < 0 or x < 0 or y + 120 > self.shape[0] or x + 120 > self.shape[1]:
            raise ValueError("Window outside input grid")
        output = np.empty((120, 120), bool)
        for cy in range(y // self.chunk, (y + 119) // self.chunk + 1):
            for cx in range(x // self.chunk, (x + 119) // self.chunk + 1):
                valid = self._read_chunk(scene, cy, cx)
                y0, x0 = max(y, cy * self.chunk), max(x, cx * self.chunk)
                y1, x1 = min(y + 120, (cy + 1) * self.chunk), min(x + 120, (cx + 1) * self.chunk)
                output[y0 - y : y1 - y, x0 - x : x1 - x] = valid[
                    y0 - cy * self.chunk : y1 - cy * self.chunk,
                    x0 - cx * self.chunk : x1 - cx * self.chunk,
                ]
        return output

    def verify_full(self, demands):
        """Require full inputs and compare any downloaded masks before publication."""
        compared = 0
        for demand in demands:
            scene = demand["scene"]
            done = self.full.done(scene)
            for cy, cx in sorted(demand["chunks"]):
                if not done[cy, cx]:
                    raise FileNotFoundError(
                        f"Selected input chunk missing: {scene['asset_id']} {cy, cx}"
                    )
                y, x = cy * self.chunk, cx * self.chunk
                try:
                    # Explicit base call: verification must never fetch or prefer full inputs.
                    tile = QualityCache.get_tile(self, scene, y // self.tile, x // self.tile)
                except FileNotFoundError:
                    continue  # This selection reused an already completed full-band chunk.
                y1, x1 = min(self.shape[0], y + self.chunk), min(self.shape[1], x + self.chunk)
                expected = tile[
                    y % self.tile : y % self.tile + y1 - y, x % self.tile : x % self.tile + x1 - x
                ]
                actual = self.full._open(scene)["valid"][y:y1, x:x1].astype(bool)
                if not np.array_equal(actual, expected):
                    raise ValueError(
                        f"Selection/export mask mismatch: {scene['asset_id']} {cy, cx}"
                    )
                compared += 1
        return compared
