"""Cached scene metadata and bounded, balanced centre proposals."""

import json
import random
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from hydrostate.data.gee.client import retry
from hydrostate.data.gee.contracts import atomic_json, fingerprint, locked
from hydrostate.data.gee.products import collection
from hydrostate.data.spatial import geographic_split, grid

from .composites import scene_catalog
from .contract import CANDIDATE_VERSION
from .sampling import inside_split_block


def cached_json(path, build):
    path.parent.mkdir(parents=True, exist_ok=True)
    with locked(str(path) + ".lock"):
        if path.exists():
            return json.loads(path.read_text())
        value = retry(build)
        atomic_json(path, value)
        return value


def pml_calendar(ee, config, cache):
    def fetch():
        images = collection(ee, "pml").filterDate(
            f"{config['start_year']}-01-01", f"{config['end_year'] + 1}-01-01"
        )
        return ee.FeatureCollection(
            images.toList(images.size().max(1)).map(
                lambda im: ee.Feature(
                    None, ee.Image(im).toDictionary(["asset_id", "system:time_start"])
                )
            )
        ).getInfo()["features"]

    rows = cached_json(cache / "pml_calendar.json", fetch)
    calendar = {}
    for feature in rows:
        props = feature["properties"]
        start = datetime.fromtimestamp(props["system:time_start"] / 1000, UTC).date().isoformat()
        calendar[start] = props
    if not calendar:
        raise ValueError("No PML periods in the requested years")
    return calendar


def global_blocks(config, seed_offset=0):
    step = config["candidate_block_degrees"]
    blocks = [(x, y) for y in range(-60, 84, step) for x in range(-180, 180, step)]
    random.Random(config["candidate_seed"] + seed_offset).shuffle(blocks)
    return blocks


def centre_sample(feature, block, config):
    """Make the sampled coordinate exactly output pixel (20,20)'s centre."""
    from pyproj import Transformer

    lon, lat = feature["geometry"]["coordinates"]
    if not inside_split_block(lon, lat, config["split_margin_m"]):
        return None
    crs, _ = grid(lon, lat)
    cx, cy = Transformer.from_crs(4326, crs, always_xy=True).transform(lon, lat)
    return dict(
        feature["properties"],
        lon=lon,
        lat=lat,
        crs=crs,
        transform=[10, 0, cx - 615, 0, -10, cy + 615],
        block=block,
        split=geographic_split(dict(lon=lon, lat=lat), config["split_seed"]),
    )


class CandidateCatalog:
    """One cached JRC block per worker task; no nested thread pool."""

    def __init__(self, ee, config):
        from pyproj import CRS

        step = config["candidate_block_degrees"]
        occurrence = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").unmask(0)
        occurrence = occurrence.mask(ee.Image.constant(1))
        observed = ee.ImageCollection("JRC/GSW1_4/YearlyHistory").select("waterClass").max().gt(0)
        bins = (
            occurrence.divide(20)
            .floor()
            .add(1)
            .where(occurrence.eq(0), 0)
            .where(occurrence.eq(100), 6)
            .toInt()
            .rename("frequency_bin")
            .updateMask(observed)
        )
        climate = (
            ee.Image(config["climate_asset"]).select(0).unmask(0, False).rename("climate_class")
        )
        strata = bins.addBands(climate).addBands(occurrence.rename("occurrence"))
        contract = {
            k: config[k]
            for k in (
                "candidate_seed",
                "candidate_block_degrees",
                "candidates_per_bin",
                "climate_asset",
                "split_margin_m",
            )
        }
        cache = Path(config["candidate_dir"]) / fingerprint(
            dict(contract, version=CANDIDATE_VERSION)
        )
        projection = CRS.from_epsg(6933).to_wkt(version="WKT1_GDAL")

        self.ee, self.config, self.step = ee, config, step
        self.cache, self.strata, self.projection = cache, strata, projection

    def fetch(self, item):
        ee, config, step = self.ee, self.config, self.step
        cache, strata, projection = self.cache, self.strata, self.projection
        i, (x, y) = item
        area = ee.Geometry.Rectangle([x, y, x + step, min(y + step, 84)], geodesic=False)
        rows = cached_json(
            cache / f"{x}_{y}.json",
            lambda: strata.stratifiedSample(
                numPoints=config["candidates_per_bin"],
                classBand="frequency_bin",
                region=area,
                projection=projection,
                scale=30,
                seed=config["candidate_seed"] + i,
                geometries=True,
                tileScale=4,
            ).getInfo()["features"],
        )
        result = []
        for feature in rows:
            if sample := centre_sample(feature, [x, y, step], config):
                result.append(
                    dict(sample, sampling_group=sample["frequency_bin"], jrc_observed=True)
                )
        return result


def catalog(ee, cache, sensor, sample, start, end):
    # Same block/window requests share metadata; filter to the tile on the server
    # during pixel extraction. Scene footprint intersections are checked locally.
    x, y, step = sample["block"]
    key = fingerprint([sensor, x, y, step, start, end])
    area = ee.Geometry.Rectangle([x, y, x + step, min(y + step, 84)], geodesic=False)
    rows = cached_json(
        cache / "catalog" / f"{key}.json", lambda: scene_catalog(ee, sensor, area, start, end)
    )
    from pyproj import Transformer
    from shapely.geometry import Polygon, shape

    tr = sample["transform"]
    to_lonlat = Transformer.from_crs(sample["crs"], 4326, always_xy=True)
    tile = Polygon(
        [
            to_lonlat.transform(tr[2] + dx, tr[5] + dy)
            for dx, dy in ((0, 0), (1200, 0), (1200, -1200), (0, -1200))
        ]
    )
    # Coverage labels also screen the larger SMAP footprint around this tile.
    footprint = (
        tile.buffer(0.35 / max(0.2, np.cos(np.deg2rad(sample["lat"])))) if sensor == "dw" else tile
    )
    return [row["properties"] for row in rows if shape(row["geometry"]).intersects(footprint)]
