"""Offline scene Zarr storage. No network requests are made by this module."""

from __future__ import annotations

import math
import shutil
from collections import OrderedDict
from pathlib import Path
from threading import Event, RLock

import numpy as np
import zarr
from numcodecs import Blosc

from hydrostate.data.gee.contracts import fingerprint, input_shape, locked
from hydrostate.data.gee.products import normalize_pixels
from hydrostate.data.input_products import L2_PROFILE, LEGACY_PROFILE, profile_bands


class SceneCache:
    """One ordinary Zarr group per scene; only requested spatial chunks exist.

    A scene lock serializes its writes. The done bitmap is written last, so an
    interrupted download is safely repeated. Inference is strictly read-only.
    """

    def __init__(self, root, plan, *, min_free_gib=5):
        self.root, self.plan = Path(root), plan
        if not math.isfinite(min_free_gib) or min_free_gib < 5:
            raise ValueError("min_free_gib must be finite and at least 5")
        self.min_free_gib = min_free_gib
        self.stopped = Event()
        self.grid, self.chunk = plan["grid"], plan["config"]["cache_chunk"]
        if not 120 <= self.chunk <= 720 or self.chunk % 3:
            raise ValueError("cache_chunk must be a multiple of 3 in [120,720]")
        from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.data.normalize import (
            load_computed_config,
        )

        self.norm = load_computed_config()
        self.profile = plan['config'].get('input_profile', LEGACY_PROFILE)
        self.bands = profile_bands(self.profile)
        self._groups = OrderedDict()
        self._groups_lock = RLock()
        self.contract = fingerprint(
            dict(
                grid=self.grid,
                normalization=self.norm,
                bands=self.bands,
                preprocessing=("gee_sampling.prepare_v1" if self.profile == LEGACY_PROFILE
                               else self.profile),
            )
        )

    def path(self, scene):
        return self.root / "input_cache" / scene["key"] / (fingerprint(scene["asset_id"]) + ".zarr")

    def _open(self, scene, writable=False):
        with self._groups_lock:
            return self._open_group(scene, writable)

    def _open_group(self, scene, writable):
        path = self.path(scene)
        if not writable:
            if str(path) in self._groups:
                self._groups.move_to_end(str(path))
                return self._groups[str(path)]
            group = zarr.open_group(str(path), mode="r")
        else:
            group = zarr.open_group(str(path), mode="a")
            if "contract" not in group.attrs:
                h, w = input_shape(self.grid)
                bands, chunk = len(self.bands[scene["key"]]), self.chunk
                compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
                group.require_dataset(
                    "values",
                    shape=(bands, h, w),
                    chunks=(bands, chunk, chunk),
                    dtype=("f4" if self.profile == L2_PROFILE and scene['key'] == 'landsat'
                           else "f2"),
                    compressor=compressor,
                )
                group.require_dataset(
                    "valid", shape=(h, w), chunks=(chunk, chunk), dtype="u1", compressor=compressor
                )
                done_shape = (math.ceil(h / chunk), math.ceil(w / chunk))
                group.require_dataset(
                    "done", shape=done_shape, chunks=done_shape, dtype="u1", compressor=compressor
                )
                group.attrs.update(contract=self.contract, scene=scene)
        if group["values"].chunks[1:] != (self.chunk, self.chunk):
            raise ValueError("Scene cache chunk layout mismatch")
        if group.attrs["contract"] != self.contract:
            raise ValueError("Scene cache preprocessing/grid mismatch")
        if not writable:
            self._groups[str(path)] = group
            if len(self._groups) > 64:
                self._groups.popitem(last=False)
        return group

    def read(self, scene, window):
        if self.stopped.is_set():
            raise RuntimeError("Input preparation stopped after a worker failure")
        y, x = window["row"] * 3, window["col"] * 3
        indices = [
            (cy, cx)
            for cy in range(y // self.chunk, (y + 119) // self.chunk + 1)
            for cx in range(x // self.chunk, (x + 119) // self.chunk + 1)
        ]
        group = self._open(scene)
        if any(not group["done"][cy, cx] for cy, cx in indices):
            raise FileNotFoundError(f"Incomplete cached scene: {scene['asset_id']}")
        return group["values"][:, y : y + 120, x : x + 120], group["valid"][
            y : y + 120, x : x + 120
        ].astype(bool)

    def read_valid(self, scene, window):
        group = self._open(scene)
        y, x = window["row"] * 3, window["col"] * 3
        for cy in range(y // self.chunk, (y + 119) // self.chunk + 1):
            for cx in range(x // self.chunk, (x + 119) // self.chunk + 1):
                if not group["done"][cy, cx]:
                    raise FileNotFoundError(f"Incomplete cached scene: {scene['asset_id']}")
        return group["valid"][y : y + 120, x : x + 120].astype(bool)

    def done(self, scene):
        if not self.path(scene).exists():
            h, w = input_shape(self.grid)
            return np.zeros((math.ceil(h / self.chunk), math.ceil(w / self.chunk)), bool)
        return self._open(scene)["done"][:].astype(bool)

    def write_chunk(self, scene, cy, cx, values, valid):
        if shutil.disk_usage(self.root).free < self.min_free_gib * 1024**3:
            raise OSError(f"Less than {self.min_free_gib:g} GiB free")
        h, w = input_shape(self.grid)
        y, x = cy * self.chunk, cx * self.chunk
        expected = (min(self.chunk, h - y), min(self.chunk, w - x))
        if valid.shape != expected or values.shape != (len(self.bands[scene["key"]]), *expected):
            raise ValueError("Scene chunk shape does not match its grid")
        if self.profile == L2_PROFILE and scene['key'] == 'landsat':
            normalized = np.where(valid[None], values, 0).astype('float32')
            if not np.isfinite(normalized).all():
                raise ValueError('Nonfinite L2 reflectance')
        else:
            normalized = normalize_pixels(scene["key"], values, valid, self.norm)
        with locked(str(self.path(scene)) + ".lock"):
            group = self._open(scene, writable=True)
            if group["done"][cy, cx]:
                return
            group["values"][:, y : y + expected[0], x : x + expected[1]] = normalized
            group["valid"][y : y + expected[0], x : x + expected[1]] = valid
            if not np.array_equal(
                group["values"][:, y : y + expected[0], x : x + expected[1]], normalized
            ) or not np.array_equal(
                group["valid"][y : y + expected[0], x : x + expected[1]], valid
            ):
                raise OSError("Scene cache readback failed")
            # Flush the actual Zarr chunk files before publishing the done marker.
            self._sync_chunk(group["values"], (0, cy, cx))
            self._sync_chunk(group["valid"], (cy, cx))
            group["done"][cy, cx] = 1
            self._sync_chunk(group["done"], (0, 0))

    @staticmethod
    def _sync_chunk(array, coordinates):
        import os

        path = Path(array.chunk_store.path) / array._chunk_key(coordinates)
        with path.open("rb") as handle:
            os.fsync(handle.fileno())
