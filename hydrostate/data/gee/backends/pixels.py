"""Direct pixel transport for small requests and the legacy regional reader."""

import shutil

import numpy as np

from hydrostate.data.gee.client import retry
from hydrostate.data.gee.contracts import input_shape, input_transform, locked, map_point
from hydrostate.data.gee.products import BANDS, normalize_pixels, scene_expression
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
