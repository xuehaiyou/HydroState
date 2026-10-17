"""Direct pixel transport for small requests and the legacy regional reader."""

import shutil
import time
from concurrent.futures import Future

import numpy as np

from hydrostate.data.gee.client import retry
from hydrostate.data.gee.contracts import input_shape, input_transform, locked, map_point
from hydrostate.data.gee.products import (
    BANDS,
    normalize_pixels,
    scene_expression,
    validity_expression,
)
from hydrostate.storage.quality_cache import QualityCache
from hydrostate.storage.scene_cache import SceneCache


def compute_pixels(ee, image, names, crs, transform, width, height):
    """Return named, finite bands on exactly the requested pixel grid."""
    raw = retry(
        lambda: ee.data.computePixels(
            dict(
                expression=image,
                fileFormat="NUMPY_NDARRAY",
                bandIds=list(names),
                grid=dict(
                    dimensions=dict(width=width, height=height),
                    crsCode=crs,
                    affineTransform=dict(
                        zip(
                            ("scaleX", "shearX", "translateX", "shearY", "scaleY", "translateY"),
                            transform,
                        )
                    ),
                ),
            )
        )
    )
    if raw.shape != (height, width) or list(raw.dtype.names or ()) != list(names):
        raise ValueError("Unexpected computePixels shape/band order")
    if any(not np.isfinite(raw[name]).all() for name in names):
        raise ValueError("Nonfinite GEE response")
    return raw


class PixelQualityCache(QualityCache):
    """Deduplicated mask-only requests; independent tiles download concurrently."""

    def __init__(self, full_cache, *, ee, **kwargs):
        super().__init__(full_cache, **kwargs)
        self.ee = ee
        self._inflight = {}
        self.requests = self.response_bytes = 0
        self.request_seconds = 0.0

    def get_tile(self, scene, ty, tx):
        key = (scene["asset_id"], ty, tx)
        with self.guard:
            try:
                return super().get_tile(scene, ty, tx)
            except FileNotFoundError:
                pass
            owner = key not in self._inflight
            if owner:
                self._inflight[key] = Future()
            future = self._inflight[key]
        if not owner:
            return future.result()
        try:
            if self.stopped.is_set():
                raise RuntimeError("Input preparation stopped after a worker failure")
            if shutil.disk_usage(self.root).free < self.full.min_free_gib * 1024**3:
                raise OSError(f"Less than {self.full.min_free_gib:g} GiB free")
            y, x, y1, x1 = self.tile_bounds(ty, tx)
            left, top = map_point(input_transform(self.grid), x, y)
            started = time.monotonic()
            raw = compute_pixels(
                self.ee,
                validity_expression(self.ee, scene),
                ["valid"],
                self.grid["crs"],
                [10, 0, left, 0, -10, top],
                x1 - x,
                y1 - y,
            )
            valid = self.write_tile(scene, ty, tx, raw["valid"])
            with self.guard:
                self.requests += 1
                self.response_bytes += raw.nbytes
                self.request_seconds += time.monotonic() - started
            future.set_result(valid)
            return valid
        except BaseException as exc:
            future.set_exception(exc)
            raise
        finally:
            with self.guard:
                self._inflight.pop(key, None)


class PixelSceneCache(SceneCache):
    """Compatibility adapter; the underlying storage remains network independent."""

    def __init__(self, root, plan, *, ee=None, min_free_gib=5):
        super().__init__(root, plan, min_free_gib=min_free_gib)
        self.ee = ee

    def _download(self, scene, cy, cx, group):
        if self.stopped.is_set():
            raise RuntimeError("Input preparation stopped after a worker failure")
        if shutil.disk_usage(self.root).free < self.min_free_gib * 1024**3:
            raise OSError(f"Less than {self.min_free_gib:g} GiB free")
        if self.ee is None:
            raise FileNotFoundError("Scene chunk is not prepared")
        h, w = input_shape(self.grid)
        y, x = cy * self.chunk, cx * self.chunk
        height, width = min(self.chunk, h - y), min(self.chunk, w - x)
        left, top = map_point(input_transform(self.grid), x, y)
        names = [*BANDS[scene["key"]], "valid"]
        raw = compute_pixels(
            self.ee,
            scene_expression(self.ee, scene),
            names,
            self.grid["crs"],
            [10, 0, left, 0, -10, top],
            width,
            height,
        )
        valid = raw["valid"] > 0
        values = np.stack([raw[b] for b in names[:-1]])
        group["values"][:, y : y + height, x : x + width] = normalize_pixels(
            scene["key"], values, valid, self.norm
        )
        group["valid"][y : y + height, x : x + width] = valid
        group["done"][cy, cx] = 1

    def _ensure(self, scene, window):
        if self.stopped.is_set():
            raise RuntimeError("Input preparation stopped after a worker failure")
        if self.ee is None:
            return
        y, x = window["row"] * 3, window["col"] * 3
        with locked(str(self.path(scene)) + ".lock"):
            group = self._open(scene, writable=True)
            for cy in range(y // self.chunk, (y + 119) // self.chunk + 1):
                for cx in range(x // self.chunk, (x + 119) // self.chunk + 1):
                    if not group["done"][cy, cx]:
                        self._download(scene, cy, cx, group)

    def read(self, scene, window):
        self._ensure(scene, window)
        return super().read(scene, window)

    def read_valid(self, scene, window):
        self._ensure(scene, window)
        return super().read_valid(scene, window)
