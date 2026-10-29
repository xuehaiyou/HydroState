"""Atomic, independently reusable slot/block stores and verified GeoTIFF ingestion."""

import json
import os
import shutil
from pathlib import Path

import numpy as np
import zarr
from affine import Affine
from numcodecs import Blosc

from hydrostate.data.gee.contracts import atomic_json
from hydrostate.data.regional.plan import block_grid
from hydrostate.data.retrieval.composites import QUALITY, normalize
from hydrostate.data.retrieval.contract import INPUT_GROUPS


def product_path(root, item):
    return Path(root) / "inputs" / item["id"][:2] / (item["id"] + ".zarr")


def ready(root, item):
    path = product_path(root, item) / "_SUCCESS.json"
    if not path.exists():
        return False
    if json.loads(path.read_text()) != item:
        raise ValueError(f"Regional product identity mismatch: {path}")
    return True


def normalize_raw(plan, name, raw):
    """Normalize a full block or aligned subrectangle exactly once."""
    spec = INPUT_GROUPS[name]
    if (raw.ndim != 3 or raw.shape[0] != len(spec.bands) + len(QUALITY)
            or not np.isfinite(raw).all()):
        raise ValueError("Invalid exported array shape or nonfinite values")
    quality = raw[len(spec.bands) :]
    days = quality[0]
    valid = days > 0
    if (
        np.any((days < 0) | (days > 8) | (days != np.round(days)))
        or np.any(quality[1:, valid] < 0)
        or np.any(quality[1:, valid] >= 8)
        or np.any(quality[2, valid] > quality[1, valid])
        or np.any(quality[1, valid] > quality[3, valid])
    ):
        raise ValueError("Invalid date counts or temporal support in regional composite")
    return dict(
        values=normalize(
            spec.sensor, raw[: len(spec.bands)], valid, plan["normalization"], spec.bands
        ),
        valid=valid[None].astype("u1"),
        quality=np.where(valid[None], quality, 0).astype("f2"),
    )


def commit_arrays(root, plan, item, arrays, *, reuse=None):
    """Publish complete normalized arrays, including verified reused pixel rectangles."""
    if ready(root, item):
        return
    spec = INPUT_GROUPS[item["group"]]
    grid = block_grid(plan, item["block"], spec.resolution)
    channels = dict(values=len(spec.bands), valid=1, quality=len(QUALITY))
    if set(arrays) != set(channels) or any(
        arrays[k].shape != (c, grid["height"], grid["width"])
        or not np.isfinite(arrays[k]).all() for k, c in channels.items()
    ):
        raise ValueError("Invalid normalized regional arrays")
    if not np.isin(arrays["valid"], [0, 1]).all():
        raise ValueError("Regional validity must be binary")
    path = product_path(root, item)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".zarr.part")
    if temp.exists():
        shutil.rmtree(temp)  # Only this product's uncommitted staging store.
    group = zarr.open_group(str(temp), mode="w")
    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    for name, value in arrays.items():
        group.create_dataset(
            name,
            data=value,
            chunks=(value.shape[0], min(256, grid["height"]), min(256, grid["width"])),
            compressor=compressor,
        )
    group.attrs.update(identity=item, grid=grid, input_contract=plan["input_contract"])
    if reuse:
        atomic_json(temp / "_REUSE.json", reuse)
    atomic_json(temp / "_SUCCESS.json", item)
    os.replace(temp, path)


def commit(root, plan, item, raw):
    """Normalize exactly once, as in training; invalid zeros remain masked."""
    if not ready(root, item):
        commit_arrays(root, plan, item, normalize_raw(plan, item["group"], raw))


def ingest(root, plan, job, paths):
    """Allow only aligned one-pixel outer rounding; reject holes and overlaps."""
    import rasterio

    grid, products = job["grid"], job["products"]
    h, w = grid["height"], grid["width"]
    expected = Affine(*grid["transform"])
    values = np.empty((len(job["bands"]), h, w), "f4")
    covered = np.zeros((h, w), bool)
    for path in paths:
        with rasterio.open(path) as src:
            relative = ~expected * src.transform
            if (
                src.crs != rasterio.crs.CRS.from_user_input(grid["crs"])
                or list(src.descriptions) != job["bands"]
                or not np.allclose(
                    [relative.a, relative.b, relative.d, relative.e],
                    [1, 0, 0, 1],
                    atol=1e-7,
                    rtol=0,
                )
            ):
                raise ValueError("Export CRS, pixel spacing or bands differ from plan")
            x, y = round(relative.c), round(relative.f)
            if max(abs(x - relative.c), abs(y - relative.f)) > 1e-5:
                raise ValueError("Export origin is not pixel aligned")
            if min(x, y) < -1 or x + src.width > w + 1 or y + src.height > h + 1:
                raise ValueError("Export exceeds the planned bounds")
            x0, y0, x1, y1 = max(x, 0), max(y, 0), min(w, x + src.width), min(h, y + src.height)
            if x1 <= x0 or y1 <= y0:
                raise ValueError("Export does not intersect the requested interior")
            if covered[y0:y1, x0:x1].any():
                raise ValueError("Duplicate or overlapping export tiles")
            window = rasterio.windows.Window(x0 - x, y0 - y, x1 - x0, y1 - y0)
            if not src.read_masks(window=window).all():
                raise ValueError("Missing export pixels; empty observations must use explicit QA")
            values[:, y0:y1, x0:x1] = src.read(window=window)
            covered[y0:y1, x0:x1] = True
    if not covered.all():
        raise ValueError("Export tiles do not cover the planned block")
    n = len(job["bands"]) // len(products)
    for index, item in enumerate(products):
        commit(root, plan, item, values[index * n : (index + 1) * n])
