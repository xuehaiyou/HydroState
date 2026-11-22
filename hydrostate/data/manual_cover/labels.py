"""Point deduplication and geometric, not nearest-neighbour, raster aggregation."""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import rasterio
import shapely
from affine import Affine
from pyproj import Transformer
from rasterio.features import shapes
from shapely.geometry import shape
from shapely.ops import transform

from .contract import CLASSES, FULL_AREA_TOLERANCE, ManualCoverSchema


def empty_targets():
    return {
        k: np.zeros(s.tail_shape, dtype=s.dtype) for k, s in ManualCoverSchema().targets.items()
    }


def point_fractions(points):
    result = empty_targets()
    for (r, c), group in points.groupby(["output_row", "output_col"]):
        if not (0 <= r < 40 and 0 <= c < 40):
            raise ValueError("Point outside output grid")
        result["point_count"][0, r, c] = len(group)
        classes = set(group.cover_class)
        if not classes <= {0, 1, 2}:
            raise ValueError("Invalid point class")
        if len(classes) != 1:
            result["point_conflict"][0, r, c] = 1
            continue
        cls = classes.pop()
        for k, name in enumerate(CLASSES):
            result[name][0, r, c] = float(k == cls)
            result[name + "_valid"][0, r, c] = 1
    # Points do not measure a valid annotated area. Leave valid_area_fraction zero.
    return result


def raster_regions(path, target_crs):
    """Union original pixel polygons before intersection, preserving holes and nodata."""
    with rasterio.open(path) as src:
        values = src.read(1)
        if src.count != 1 or not np.isin(values, [-1, 0, 1]).all():
            raise ValueError(f"Not a manual binary water label: {path}")
        project = Transformer.from_crs(src.crs, target_crs, always_xy=True).transform
        regions = {}
        for name, mask in (("water", values == 1), ("nonwater", values == 0)):
            polygons = [
                transform(project, shape(g))
                for g, v in shapes(mask.astype("uint8"), mask=mask, transform=src.transform)
                if v
            ]
            regions[name] = shapely.union_all(polygons)
        return regions


@lru_cache(maxsize=3)
def event_regions(root, event, crs):
    table = json.loads((Path(root) / "sources" / "flood_catalog.json").read_text())
    regions = [
        raster_regions(Path(root) / r["path"], crs)
        for r in table
        if r["event"] == event and r["included"]
    ]
    if not regions:
        raise ValueError(f"No manual labels for {event}")
    water = shapely.union_all([r["water"] for r in regions])
    nonwater = shapely.union_all([r["nonwater"] for r in regions])
    # Overlapping chips with conflicting annotations have no usable truth.
    conflict = water.intersection(nonwater)
    return water.difference(conflict), water.union(nonwater).difference(conflict)


def area_fractions(water, valid, reference):
    affine = Affine(*reference) * Affine.scale(3)
    if affine.b or affine.d or affine.a != 30 or affine.e != -30:
        raise ValueError("Expected north-up 30 m output grid")
    rr, cc = np.indices((40, 40))
    x = affine.c + cc.ravel() * 30
    y = affine.f - rr.ravel() * 30
    cells = shapely.box(x, y - 30, x + 30, y)
    extent = shapely.box(x.min(), y.min() - 30, x.max() + 30, y.max())
    valid = valid.intersection(extent)
    water = water.intersection(valid)
    area = shapely.area(shapely.intersection(cells, valid)).reshape(40, 40)
    wet = shapely.area(shapely.intersection(cells, water)).reshape(40, 40)
    result = empty_targets()
    result["valid_area_fraction"][0] = np.clip(area / 900, 0, 1)
    result["water"][0] = np.divide(wet, area, out=np.zeros_like(wet), where=area > 0)
    result["water_valid"][0] = area / 900 >= 1 - FULL_AREA_TOLERANCE
    return result
