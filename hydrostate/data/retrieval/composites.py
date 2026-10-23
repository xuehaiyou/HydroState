"""Server-side QA, daily deduplication and date-weighted composites."""

import numpy as np

from hydrostate.data.gee.products import BANDS, MODALITIES, collection

from .specs import INPUT_SPECS

DW_CLASSES = [
    "water",
    "trees",
    "grass",
    "flooded_vegetation",
    "crops",
    "shrub_and_scrub",
    "built",
    "bare",
    "snow_and_ice",
]
QUALITY = ["days", "mean_day", "first_day", "last_day"]


def optical_mask(image, sensor):
    qa = INPUT_SPECS[sensor].qa
    if qa == "landsat_c2_clear_water_snow":
        # Bits 0..4: fill, dilated cloud, cirrus, cloud, shadow. Snow (5) stays.
        return image.select("QA_PIXEL").bitwiseAnd(31).eq(0).And(image.select("QA_RADSAT").eq(0))
    if qa != "scl_clear_water_snow":
        raise ValueError(f"No optical QA adapter for {qa}")
    scl = image.select("SCL")
    return scl.remap([0, 1, 3, 8, 9, 10], [0] * 6, 1).eq(1)


def prepare_sensor(ee, image, sensor, bands=None):
    bands = list(bands or BANDS[sensor])
    spec = INPUT_SPECS[sensor]
    if spec.normalization == "toa_physical":
        value = ee.Algorithms.Landsat.TOA(image).select(bands)
        value = value.updateMask(optical_mask(image, sensor))
    elif spec.qa == "scl_clear_water_snow":
        value = image.select(bands).updateMask(optical_mask(image, sensor))
    elif spec.qa == "grd_iw_valid":
        # GRD catalog values are calibrated dB; average linear power, not dB.
        value = ee.Image.constant(10).pow(image.select(bands).divide(10))
    else:
        raise ValueError(f"No input preparation adapter for {sensor}: {spec.qa}")
    valid = value.mask().reduce(ee.Reducer.min()).gt(0)
    return ee.Image(
        value.updateMask(valid).copyProperties(image, ["system:time_start", "system:index"])
    )


def daily_collection(ee, images):
    """Overlapping granules on one UTC day have one vote, not one vote each."""
    images = images.map(lambda im: im.toFloat().set("day", im.date().format("YYYY-MM-dd")))
    days = ee.List(images.aggregate_array("day")).distinct().sort()
    return ee.ImageCollection.fromImages(
        days.map(
            lambda day: (
                images.filter(ee.Filter.eq("day", day))
                .mean()
                .toFloat()
                .set("system:time_start", ee.Date(day).millis())
            )
        )
    )


def mean_composite(ee, images, bands, start, *, radar=False):
    daily = daily_collection(ee, images)
    empty = ee.Image.constant([0] * len(bands)).rename(bands).toFloat().updateMask(ee.Image(0))
    # An empty, masked image establishes names without adding an observation.
    daily = daily.merge(
        ee.ImageCollection([empty.set("system:time_start", ee.Date(start).millis())])
    )
    value = daily.select(bands).mean()
    count = daily.select(bands[0]).count().rename("days").unmask(0, False)

    def observation_day(im):
        day = ee.Image.constant(im.date().difference(ee.Date(start), "day"))
        return day.rename("day").toFloat().updateMask(im.select(bands[0]).mask())

    dates = daily.map(observation_day)
    stats = ee.Image.cat(
        [
            count,
            dates.mean().rename("mean_day"),
            dates.min().rename("first_day"),
            dates.max().rename("last_day"),
        ]
    )
    if radar:
        value = value.log10().multiply(10)
    return ee.Image.cat([value, stats]).unmask(0, False).toFloat()


def sensor_composite(ee, sensor, scene_ids, start, bands=None):
    bands = list(bands or BANDS[sensor])
    images = ee.ImageCollection.fromImages([ee.Image(sid) for sid in scene_ids])
    return mean_composite(
        ee,
        images.map(lambda im: prepare_sensor(ee, im, sensor, bands)),
        bands,
        start,
        radar=INPUT_SPECS[sensor].aggregation == "linear_power_daily_mean",
    )


def prepare_cover(ee, image, thresholds):
    probabilities = image.select(DW_CLASSES)
    # Argmax gives one mutually exclusive class, including deterministic ties.
    winner = probabilities.toArray().arrayArgmax().arrayGet([0])
    pw, pi = image.select("water"), image.select("snow_and_ice")
    pl = probabilities.select(DW_CLASSES[1:8]).reduce(ee.Reducer.sum())
    water = winner.eq(0).And(pw.gte(thresholds["water"]))
    ice = pi.gte(thresholds["snow_ice"])
    land = (
        pl.gte(thresholds["land"])
        .And(pw.lt(thresholds["land_max_water_ice"]))
        .And(pi.lt(thresholds["land_max_water_ice"]))
    )
    known = water.Or(ice).Or(land)
    observed = probabilities.mask().reduce(ee.Reducer.min()).gt(0)
    return ee.Image(
        ee.Image.cat([water.rename("water"), ice.rename("snow_ice"), land.rename("land")])
        .updateMask(known)
        .addBands(known.Not().rename("unknown").updateMask(observed))
        .toFloat()
        .copyProperties(image, ["system:time_start"])
    )


def cover_composite(ee, scene_ids, start, thresholds):
    images = ee.ImageCollection.fromImages([ee.Image(sid) for sid in scene_ids]).map(
        lambda im: prepare_cover(ee, im, thresholds)
    )
    composite = mean_composite(ee, images, ["water", "snow_ice", "land"], start)
    known = composite.select("days").gt(0).rename("known")
    # Count valid observations even when classification is Unknown; clouds and
    # missing pixels have no vote. Unknown is QA, not a fourth surface class.
    quality = mean_composite(ee, images, ["unknown"], start).select(
        ["unknown", "days"], ["unknown_fraction", "observed_days"]
    )
    return composite.addBands(known).addBands(quality)


def scene_catalog(ee, sensor, area, start, end):
    """One metadata request per sensor and spatial/date block; no per-scene getInfo."""
    images = collection(ee, sensor).filterBounds(area).filterDate(start, end)
    return ee.FeatureCollection(
        images.toList(images.size().max(1)).map(
            lambda image: ee.Feature(
                ee.Image(image).geometry().bounds(),
                ee.Image(image).toDictionary(
                    [
                        "asset_id",
                        "system:time_start",
                        "relativeOrbitNumber_start",
                        "orbitProperties_pass",
                    ]
                ),
            )
        )
    ).getInfo()["features"]


def normalize(sensor, values, valid, norm, bands=None):
    """Calibrated L1 has explicit physical scaling and a trainable input adapter.

    S1/S2 retain the published pretraining normalization. TOA reflective channels
    use reflectance / 2 + .5; B10/B11 use (kelvin - 290) / 80 + .5.
    These are physical scales, not purported pretrained DN statistics.
    """
    values = values.astype(np.float32)
    bands = list(bands or BANDS[sensor])
    spec = INPUT_SPECS[sensor]
    band_specs = {band.name: band for band in spec.bands}
    if spec.normalization == "toa_physical":
        for index, band in enumerate(bands):
            definition = band_specs[band]
            values[index] = (values[index] - definition.offset) / definition.divisor + 0.5
    elif spec.normalization == "pretrained":
        for index, band in enumerate(bands):
            name = band_specs[band].normalization_key
            stats = norm[MODALITIES[sensor]][name]
            values[index] = (values[index] - stats["mean"]) / (4 * stats["std"]) + 0.5
    else:
        raise ValueError(f"No normalization adapter for {spec.normalization}")
    return np.where(valid[None], values, 0).astype(np.float16)
