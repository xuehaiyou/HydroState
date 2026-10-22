"""Fetch product labels and explicit support quality, separately from imagery."""

from hydrostate.data.gee.backends.pixels import compute_pixels
from hydrostate.data.gee.products import collection

from .composites import QUALITY, cover_composite
from .contract import COVER, INPUT_SIZE


def pixels(ee, image, names, sample):
    return compute_pixels(
        ee, image, names, sample["crs"], sample["transform"], INPUT_SIZE, INPUT_SIZE
    )


def cover_quality(ee, cover, geometry):
    # Fractions are unmasked before reduction: denominator is the *whole* area.
    quality = ee.Image.cat(
        [
            cover.select("known"),
            cover.select("water").add(cover.select("snow_ice")).rename("water_ice"),
        ]
    ).unmask(0, False)
    # Evaluate the confidence mask at 10 m, then preserve all known/unknown area
    # in the 30 m reduction; do not threshold probabilities after downsampling.
    quality = quality.reduceResolution(ee.Reducer.mean(), maxPixels=1024).reproject(
        crs=cover.projection(), scale=30
    )
    area = ee.Image.pixelArea()
    result = (
        quality.multiply(area)
        .addBands(area.rename("area"))
        .reduceRegion(ee.Reducer.sum(), geometry, scale=30, maxPixels=10000000, tileScale=4)
        .getInfo()
    )
    total = result.get("area") or 0
    return {key: (result.get(key) or 0) / total if total else 0 for key in ("known", "water_ice")}


def smap_label(ee, sample, cover, start, end):
    point = ee.Geometry.Point([sample["lon"], sample["lat"]])
    images = collection(ee, "sm").filterDate(start, end).filterBounds(point)

    def prepare(image):
        value = image.select("soil_moisture_am")
        valid = image.select("retrieval_qual_flag_am").eq(0).And(value.gte(0)).And(value.lte(1))
        return ee.Image(value.updateMask(valid).copyProperties(image, ["system:time_start"]))

    # Use the actual GEE publication grid (v005 is geographic), not the fine
    # tile grid or an assumed original EASE-Grid projection.
    if images.size().getInfo() == 0:
        return {"valid_days": 0}
    projection = images.first().select("soil_moisture_am").projection()
    grid = projection.getInfo()
    import math

    from affine import Affine
    from pyproj import Transformer

    affine = Affine(*grid["transform"])
    x, y = Transformer.from_crs(4326, grid["crs"], always_xy=True).transform(
        sample["lon"], sample["lat"]
    )
    col, row = (math.floor(v) for v in (~affine * (x, y)))
    left, top = affine * (col, row)
    right, bottom = affine * (col + 1, row + 1)
    cell = ee.Geometry.Rectangle([left, bottom, right, top], grid["crs"], False)
    centre = ee.Geometry.Point(affine * (col + 0.5, row + 0.5), grid["crs"])

    def observation(image):
        image = ee.Image(image)
        value = prepare(image).reduceRegion(
            ee.Reducer.first(), centre, crs=projection, scale=projection.nominalScale()
        )
        return ee.Feature(None, value).set(
            {"date": image.date().format("YYYY-MM-dd"), "asset_id": image.get("asset_id")}
        )

    observed = ee.FeatureCollection(images.toList(images.size()).map(observation)).getInfo()[
        "features"
    ]
    from collections import defaultdict

    daily = defaultdict(list)
    for feature in observed:
        item = feature["properties"]
        if item.get("soil_moisture_am") is not None:
            daily[item["date"]].append(item["soil_moisture_am"])
    result = dict(
        valid_days=len(daily),
        valid_dates=sorted(daily),
        observations=[feature["properties"] for feature in observed],
    )
    if not daily:
        return result
    result["value"] = sum(sum(v) / len(v) for v in daily.values()) / len(daily)
    result["grid"] = cover_quality(ee, cover, cell)
    # Explicit approximate footprint: quality screening, NOT a claim of exact
    # SMAP antenna response or scale-matched tile supervision.
    result["footprint"] = cover_quality(ee, cover, centre.buffer(16500))
    result["support"] = dict(grid=grid, cell=[col, row], footprint="33km_diameter_disk_approx")
    # Conservative terrain proxy for local representativeness; configurable
    # weighting is stored separately from the binary supervision mask.
    elevation = ee.ImageCollection("COPERNICUS/DEM/GLO30_2024_1").select("DEM").mosaic()
    terrain = (
        elevation.reduceRegion(ee.Reducer.stdDev(), cell, scale=90, maxPixels=100000, tileScale=4)
        .getInfo()
        .get("DEM")
    )
    result["terrain_std_m"] = terrain
    result["representativeness"] = 1.0 if terrain is None else 1 / (1 + (terrain / 100.0) ** 2)
    return result


def fetch_cover(ee, sample, config, dw_ids):
    start = sample["target_time"]
    thresholds = dict(
        water=config["water_confidence"],
        snow_ice=config["ice_confidence"],
        land=config["land_confidence"],
        land_max_water_ice=config["land_max_water_ice_probability"],
    )
    cover = cover_composite(ee, dw_ids, start, thresholds)
    # Force the composite onto the known DW 10 m projection before area pooling.
    if dw_ids:
        cover = cover.setDefaultProjection(ee.Image(dw_ids[0]).select("label").projection())
    raw = pixels(
        ee, cover, [*COVER, *QUALITY, "known", "unknown_fraction", "observed_days"], sample
    )
    return cover, {name: raw[name] for name in raw.dtype.names}


def fetch_products(ee, sample, cover, pml_scene):
    start, end = sample["target_time"], sample["target_end"]
    pml = None
    if pml_scene:
        image = ee.Image(pml_scene["asset_id"]).select(["ET", "Ew"]).multiply(0.01)
        names = ["ET", "Ew", "ET_valid", "Ew_valid"]
        masked = image.updateMask(image.gte(0))
        expression = ee.Image.cat([masked, masked.mask().gt(0).rename(names[2:])]).unmask(0, False)
        array = pixels(ee, expression, names, sample)
        pml = {name: array[name] for name in names}
        pml["projection"] = image.select("ET").projection().getInfo()
    sm = smap_label(ee, sample, cover, start, end)
    return pml, sm
