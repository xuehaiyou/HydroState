"""Canonical products, QA masks and normalization shared by both transports."""

from datetime import date, timedelta

import numpy as np

BANDS = {
    "s1": ["VV", "VH"],
    "s2": ["B2", "B3", "B4", "B8", "B5", "B6", "B7", "B8A", "B11", "B12", "B1", "B9"],
    "landsat": ["B8", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B9", "B10", "B11"],
}
COLLECTIONS = {
    "s1": "COPERNICUS/S1_GRD",
    "s2": "COPERNICUS/S2_SR_HARMONIZED",
    "dw": "GOOGLE/DYNAMICWORLD/V1",
    "pml": "projects/pml_evapotranspiration/PML/OUTPUT/PML_V22a",
}
MODALITIES = {"s1": "sentinel1", "s2": "sentinel2_l2a", "landsat": "landsat"}


def slots(target):
    t = date.fromisoformat(target)
    return [
        ((t + timedelta(days=a)).isoformat(), (t + timedelta(days=b)).isoformat())
        for a, b in [(-63, -47), (-47, -31), (-31, -15), (-15, 1)]
    ]


def prepare(ee, key, image):
    if key == "s1":
        return image.select(BANDS[key])
    if key == "s2":
        scl = image.select("SCL")
        valid = scl.neq(0).And(scl.neq(1)).And(scl.neq(3))
        for code in [8, 9, 10]:
            valid = valid.And(scl.neq(code))
        return image.select(BANDS[key]).updateMask(valid)
    if key == "landsat":
        valid = image.select("QA_PIXEL").bitwiseAnd(63).eq(0)
        valid = valid.And(image.select("QA_RADSAT").eq(0))
        return image.select(BANDS[key]).updateMask(valid)
    if key == "dw":
        return image.select("label").eq(0).rename("water")
    if key == "sm":
        sm = image.select("soil_moisture_am")
        return (
            sm.rename("soil_moisture")
            .updateMask(image.select("retrieval_qual_flag_am").eq(0))
            .updateMask(sm.gte(0).And(sm.lte(1)))
        )
    et = image.select("ET").multiply(0.01)
    return et.rename("et").updateMask(et.gte(0))


def collection(ee, key):
    if key == "landsat":
        return (
            ee.ImageCollection("LANDSAT/LC08/C02/T1")
            .map(
                lambda im: im.set(
                    "asset_id", ee.String("LANDSAT/LC08/C02/T1/").cat(im.get("system:index"))
                )
            )
            .merge(
                ee.ImageCollection("LANDSAT/LC09/C02/T1").map(
                    lambda im: im.set(
                        "asset_id", ee.String("LANDSAT/LC09/C02/T1/").cat(im.get("system:index"))
                    )
                )
            )
        )
    if key == "sm":
        return (
            ee.ImageCollection("NASA/SMAP/SPL3SMP_E/005")
            .filterDate("2015-01-01", "2023-12-04")
            .map(
                lambda im: im.set(
                    "asset_id", ee.String("NASA/SMAP/SPL3SMP_E/005/").cat(im.get("system:index"))
                )
            )
            .merge(
                ee.ImageCollection("NASA/SMAP/SPL3SMP_E/006")
                .filterDate("2023-12-04", "2100-01-01")
                .map(
                    lambda im: im.set(
                        "asset_id",
                        ee.String("NASA/SMAP/SPL3SMP_E/006/").cat(im.get("system:index")),
                    )
                )
            )
        )
    c = ee.ImageCollection(COLLECTIONS[key])
    if key == "s1":
        c = (
            c.filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.eq("resolution_meters", 10))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        )
    return c.map(
        lambda im: im.set("asset_id", ee.String(COLLECTIONS[key] + "/").cat(im.get("system:index")))
    )


def normalize_pixels(key, values, valid, norm):
    """Exactly the training ingestion normalization, including float16 rounding."""
    output = np.zeros(values.shape, dtype=np.float16)
    for index, band in enumerate(BANDS[key]):
        canonical = (
            band.lower()
            if key == "s1"
            else ("B" + band[1:].zfill(2) if key == "s2" and band != "B8A" else band)
        )
        stats = norm[MODALITIES[key]][canonical]
        value = (values[index].astype(np.float32) - stats["mean"]) / (4 * stats["std"]) + 0.5
        output[index] = np.where(valid, value, 0)
    if not np.isfinite(output).all():
        raise ValueError("Nonfinite normalized input")
    return output


def scene_expression(ee, scene):
    """Explicit validity keeps empty exports unambiguous across transports."""
    image = prepare(ee, scene["key"], ee.Image(scene["asset_id"]))
    valid = image.mask().reduce(ee.Reducer.min()).gt(0)
    return ee.Image.cat(
        [image.unmask(0, sameFootprint=False), valid.unmask(0, sameFootprint=False).rename("valid")]
    ).toFloat()


def validity_expression(ee, scene):
    """The identical all-band validity used by scene_expression, as one byte band."""
    image = prepare(ee, scene["key"], ee.Image(scene["asset_id"]))
    return (
        image.mask()
        .reduce(ee.Reducer.min())
        .gt(0)
        .unmask(0, sameFootprint=False)
        .rename("valid")
        .toUint8()
    )
