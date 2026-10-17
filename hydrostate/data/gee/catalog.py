"""Persistent scene metadata, independently reusable by both transports."""

import json
import logging
from pathlib import Path

from pyproj import Transformer
from shapely.geometry import shape
from shapely.ops import transform

from .client import retry
from .contracts import atomic_json, input_shape, input_transform
from .products import collection

LOG = logging.getLogger(__name__)


def catalog(root, plan, key, year, ee):
    dest = Path(root) / "catalogs" / f"{key}_{year}.json"
    if dest.exists():
        return json.loads(dest.read_text())
    dest.parent.mkdir(parents=True, exist_ok=True)
    grid = plan["grid"]
    h, w = input_shape(grid)
    tr = input_transform(grid)
    area = ee.Geometry.Rectangle([tr.c, tr.f - h * 10, tr.c + w * 10, tr.f], grid["crs"], False)
    images = collection(ee, key).filterBounds(area).filterDate(f"{year}-01-01", f"{year + 1}-01-01")
    properties = [
        "asset_id",
        "system:time_start",
        "relativeOrbitNumber_start",
        "orbitProperties_pass",
    ]
    features = ee.FeatureCollection(
        images.toList(images.size()).map(
            lambda im: ee.Feature(
                ee.Image(im).geometry().bounds(100), ee.Image(im).toDictionary(properties)
            )
        )
    )
    data = retry(features.getInfo)
    project = Transformer.from_crs(4326, grid["crs"], always_xy=True).transform
    scenes = []
    for feature in data["features"]:
        props = feature["properties"]
        scenes.append(
            dict(
                key=key,
                asset_id=props["asset_id"],
                time=props["system:time_start"] // 1000,
                orbit=props.get("relativeOrbitNumber_start"),
                bounds=list(transform(project, shape(feature["geometry"])).bounds),
            )
        )
    scenes.sort(key=lambda x: (x["time"], x["asset_id"]), reverse=True)
    atomic_json(dest, scenes)
    LOG.info("Catalog %s %s: %d scenes", key, year, len(scenes))
    return scenes
