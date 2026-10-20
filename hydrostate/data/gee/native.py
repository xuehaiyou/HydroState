"""Native-grid transport and local reconstruction of the existing model inputs.

Only original band values and a per-group source mask are computed in EE.
QA decisions, cross-resolution alignment and normalization happen on the HPC.
"""

import json
import math
from contextlib import ExitStack

import numpy as np
from affine import Affine

from hydrostate.data.gee.contracts import atomic_json, fingerprint, input_shape, input_transform
from hydrostate.data.gee.products import BANDS

LAYOUT = "native_v1"
QA_BANDS = {"s1": [], "s2": ["SCL"], "landsat": ["QA_PIXEL", "QA_RADSAT"]}


def band_groups(scene, info):
    """Group actual asset metadata by CRS/grid; never guess origins from scale."""
    names = [*BANDS[scene["key"]], *QA_BANDS[scene["key"]]]
    metadata = {b["id"]: b for b in info["bands"]}
    groups = {}
    dtype = "float32" if scene["key"] == "s1" else "uint16"
    for name in names:
        band = metadata[name]
        if dtype == "uint16":
            t = band["data_type"]
            if t["precision"] != "int" or t["min"] < 0 or t["max"] > 65535:
                raise ValueError(f"Unsupported optical source type: {name}")
        tr = band["crs_transform"]
        if len(tr) != 6 or not np.isfinite(tr).all() or abs(Affine(*tr).determinant) == 0:
            raise ValueError("Invalid native source transform")
        identity = fingerprint([band["crs"], tr])
        group = groups.setdefault(
            identity, dict(crs=band["crs"], source_transform=tr, bands=[], dtype=dtype)
        )
        group["bands"].append(name)
    return list(groups.values())


def source_groups(ee, root, scene):
    """Cache only projection/type metadata; this does not evaluate image pixels."""
    path = root / "catalog" / "native" / (fingerprint(scene["asset_id"]) + ".json")
    if path.exists():
        saved = json.loads(path.read_text())
        if saved["asset_id"] != scene["asset_id"] or saved["layout"] != LAYOUT:
            raise ValueError("Native metadata identity mismatch")
        return saved["groups"]
    from hydrostate.data.gee.client import retry

    names = [*BANDS[scene["key"]], *QA_BANDS[scene["key"]]]
    info = retry(lambda: ee.Image(scene["asset_id"]).select(names).getInfo())
    groups = band_groups(scene, info)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, dict(asset_id=scene["asset_id"], layout=LAYOUT, groups=groups))
    return groups


def native_job(job, groups):
    """Add bounded source-grid requests to one recoverable logical export."""
    from rasterio.warp import transform_bounds

    target = Affine(*job["transform"])
    bounds = (target.c, target.f - job["height"] * 10,
              target.c + job["width"] * 10, target.f)
    parts = []
    for group in groups:
        source = Affine(*group["source_transform"])
        left, bottom, right, top = transform_bounds(
            job["grid"]["crs"], group["crs"], *bounds, densify_pts=41
        )
        points = [(~source) * (x, y) for x in (left, right) for y in (bottom, top)]
        # A source-pixel halo covers nearest-neighbour samples at transformed edges.
        x0 = math.floor(min(p[0] for p in points)) - 1
        y0 = math.floor(min(p[1] for p in points)) - 1
        x1 = math.ceil(max(p[0] for p in points)) + 1
        y1 = math.ceil(max(p[1] for p in points)) + 1
        part = dict(
            **group, layout=LAYOUT, scene=job["scene"],
            transform=list(source * Affine.translation(x0, y0))[:6],
            width=x1 - x0, height=y1 - y0,
        )
        part["bands"] = [*group["bands"], "source_valid"]
        part["raw_bytes"] = (
            part["width"] * part["height"] * len(part["bands"]) * np.dtype(group["dtype"]).itemsize
        )
        part["id"] = fingerprint(part)[:24]
        parts.append(part)
    result = dict(job, layout=LAYOUT, parts=parts, legacy_raw_bytes=job["raw_bytes"])
    result["raw_bytes"] = sum(p["raw_bytes"] for p in parts)
    result.pop("id", None)
    result["id"] = fingerprint(result)[:24]
    return result


def native_expression(ee, part):
    image = ee.Image(part["scene"]["asset_id"]).select(part["bands"][:-1])
    # Do not infer nodata from zero: legitimate zero values must remain valid.
    valid = image.mask().reduce(ee.Reducer.min()).gt(0).rename("source_valid")
    result = ee.Image.cat([
        image.unmask(0, sameFootprint=False),
        valid.unmask(0, sameFootprint=False),
    ])
    return result.toFloat() if part["dtype"] == "float32" else result.toUint16()


def validate_part(part, rasters):
    """Reject missing, overlapping, shifted, mistyped or misidentified TIFF parts."""
    import rasterio

    from hydrostate.data.gee.backends.drive import IncompleteExport

    areas = []
    expected = Affine(*part["transform"])
    for src in rasters:
        if src.crs != rasterio.crs.CRS.from_string(part["crs"]):
            raise ValueError("Native export CRS mismatch")
        if list(src.descriptions) != part["bands"] or any(
            dtype != part["dtype"] for dtype in src.dtypes
        ):
            raise ValueError("Native export band identity/type mismatch")
        relative = ~expected * src.transform
        if not np.allclose([relative.a, relative.b, relative.d, relative.e],
                           [1, 0, 0, 1], rtol=0, atol=1e-7):
            raise ValueError("Native export grid mismatch")
        col, row = round(relative.c), round(relative.f)
        if max(abs(relative.c - col), abs(relative.f - row)) > 1e-5:
            raise ValueError("Native export origin is not aligned")
        box = (col, row, col + src.width, row + src.height)
        if min(col, row) < 0 or box[2] > part["width"] or box[3] > part["height"]:
            raise ValueError("Native export exceeds requested extent")
        if any(min(box[2], c) > max(col, a) and min(box[3], d) > max(row, b)
               for a, b, c, d in areas):
            raise ValueError("Overlapping/duplicate native export parts")
        areas.append(box)
    if sum(src.width * src.height for src in rasters) != part["width"] * part["height"]:
        raise IncompleteExport("Missing native export parts")


class NativeReader:
    """Read small destination chunks from native TIFFs, including file seams."""

    def __init__(self, job, paths):
        self.job, self.paths = job, paths
        self.stack = ExitStack()

    def __enter__(self):
        import rasterio

        try:
            self.groups = []
            for part in self.job["parts"]:
                rasters = [self.stack.enter_context(rasterio.open(p))
                           for p in self.paths.get(part["id"], [])]
                validate_part(part, rasters)
                self.groups.append((part, rasters))
            return self
        except BaseException:
            self.stack.close()
            raise

    def __exit__(self, *exc):
        return self.stack.__exit__(*exc)

    def read(self, transform, width, height):
        """Sample at destination pixel centres; zero/nodata never changes indices."""
        from pyproj import Transformer

        tr = Affine(*transform)
        yy, xx = np.indices((height, width), dtype="float64")
        xs = tr.c + tr.a * (xx + 0.5) + tr.b * (yy + 0.5)
        ys = tr.f + tr.d * (xx + 0.5) + tr.e * (yy + 0.5)
        values, valid = {}, np.ones((height, width), bool)
        for part, rasters in self.groups:
            if part["crs"] == self.job["grid"]["crs"]:
                sx, sy = xs, ys
            else:
                sx, sy = Transformer.from_crs(
                    self.job["grid"]["crs"], part["crs"], always_xy=True
                ).transform(xs, ys)
            source = ~Affine(*part["transform"])
            cols = np.floor(source.a * sx + source.b * sy + source.c).astype("int64")
            rows = np.floor(source.d * sx + source.e * sy + source.f).astype("int64")
            if (cols.min() < 0 or rows.min() < 0 or cols.max() >= part["width"]
                    or rows.max() >= part["height"]):
                raise ValueError("Native request does not cover target sample centres")
            raw = np.zeros((len(part["bands"]), height, width), dtype=part["dtype"])
            covered = np.zeros((height, width), bool)
            for src in rasters:
                from rasterio.windows import Window

                offset = ~Affine(*part["transform"]) * src.transform
                c, r = cols - round(offset.c), rows - round(offset.f)
                take = (c >= 0) & (r >= 0) & (c < src.width) & (r < src.height)
                if not take.any():
                    continue
                c0, r0, c1, r1 = c[take].min(), r[take].min(), c[take].max(), r[take].max()
                block = src.read(window=Window(int(c0), int(r0), int(c1-c0+1), int(r1-r0+1)))
                raw[:, take] = block[:, r[take] - r0, c[take] - c0]
                covered |= take
            if not covered.all():
                raise ValueError("Native files do not cover all target pixels")
            if not np.isfinite(raw).all() or not np.isin(raw[-1], [0, 1]).all():
                raise ValueError("Invalid native values/source mask")
            valid &= raw[-1] > 0
            values.update(zip(part["bands"][:-1], raw[:-1]))
        key = self.job["scene"]["key"]
        if key == "s2":
            valid &= ~np.isin(values["SCL"], [0, 1, 3, 8, 9, 10])
        elif key == "landsat":
            valid &= (values["QA_PIXEL"] & 63) == 0
            valid &= values["QA_RADSAT"] == 0
        return np.stack([values[b] for b in BANDS[key]]).astype("float32"), valid


def ingest_native(cache, job, paths):
    if (job["contract"] != cache.contract or job["chunk"] != cache.chunk
            or job["grid"] != cache.grid):
        raise ValueError("Native export does not match cache contract")
    # All source groups and file coverage are validated before any done bit is set.
    with NativeReader(job, paths) as reader:
        from hydrostate.storage.quality_cache import QualityCache

        quality = QualityCache(cache)
        h, w = input_shape(cache.grid)
        tr = input_transform(cache.grid)
        done = cache.done(job["scene"])
        for cy, cx in job["chunks"]:
            if done[cy, cx]:
                continue
            y, x = cy * cache.chunk, cx * cache.chunk
            height, width = min(cache.chunk, h-y), min(cache.chunk, w-x)
            target = list(tr * Affine.translation(x, y))[:6]
            values, valid = reader.read(target, width, height)
            try:
                tile = quality.get_tile(job["scene"], y // quality.tile, x // quality.tile)
            except FileNotFoundError:
                pass  # Older selections may have reused full inputs without mask tiles.
            else:
                expected = tile[
                    y % quality.tile:y % quality.tile + height,
                    x % quality.tile:x % quality.tile + width,
                ]
                if not np.array_equal(valid, expected):
                    raise ValueError("Native QA differs from selection mask; chunk not committed")
            cache.write_chunk(job["scene"], cy, cx, values, valid)
