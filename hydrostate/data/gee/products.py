"""Canonical products, QA masks and normalization shared by both transports."""

import numpy as np

from hydrostate.data.retrieval.specs import INPUT_SPECS, SOURCES
from hydrostate.data.temporal_sampling import observation_slots

BANDS = {name: [band.name for band in spec.bands] for name, spec in INPUT_SPECS.items()}
COLLECTIONS = {
    name: sources[0].collection for name, sources in SOURCES.items() if len(sources) == 1
}
MODALITIES = {name: spec.modality for name, spec in INPUT_SPECS.items()}


def slots(target):
    """Legacy API: existing callers retain their original historical support."""
    return observation_slots(target, {})


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
        valid = image.select("QA_PIXEL").bitwiseAnd(31).eq(0)
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
    merged = None
    source_key = INPUT_SPECS[key].source if key in INPUT_SPECS else key
    for source in SOURCES[source_key]:
        images = ee.ImageCollection(source.collection)
        if source.start:
            images = images.filterDate(source.start, source.end)
        images = images.map(
            lambda im: im.set(
                "asset_id", ee.String(source.collection + "/").cat(im.get("system:index"))
            )
        )
        merged = images if merged is None else merged.merge(images)
    if key in INPUT_SPECS and INPUT_SPECS[key].qa == "grd_iw_valid":
        merged = (
            merged.filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.eq("resolution_meters", 10))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        )
    return merged


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
