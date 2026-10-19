"""Add same-acquisition Landsat L2 SR to an immutable sample library."""

import re

import numpy as np

from hydrostate.data.gee.backends.pixels import compute_pixels
from hydrostate.data.gee_labels import decode_sample
from hydrostate.data.input_products import LANDSAT_SR_BANDS, landsat_l2_pixels

PRODUCER = "gee_landsat_l2_sr_v1"


def l2_asset(asset):
    if not re.fullmatch(r"LANDSAT/LC0[89]/C02/T1/LC0[89]_\d{6}_\d{8}", asset):
        raise ValueError(f"Expected Landsat 8/9 Collection 2 T1 L1 asset: {asset}")
    return asset.replace("/T1/", "/T1_L2/")


def matched_scenes(ee, sample):
    queries, original = {}, {}
    for slot in range(4):
        scene = sample["inputs"].get(f"landsat_{slot}")
        if scene is None:
            continue
        asset = l2_asset(scene["asset_id"])
        collection, index = asset.rsplit("/", 1)
        found = ee.ImageCollection(collection).filter(ee.Filter.eq("system:index", index))
        queries[str(slot)] = ee.Algorithms.If(
            found.size().gt(0),
            ee.Dictionary(dict(asset_id=asset,
                               time_ms=found.first().get("system:time_start"))), None)
        original[str(slot)] = scene
    metadata = ee.Dictionary(queries).getInfo() if queries else {}
    for slot, scene in metadata.items():
        if scene and abs(scene["time_ms"] - original[slot]["system:time_start"]) > 1000:
            raise ValueError("L1/L2 acquisition time mismatch")
    return metadata


def fetch_sample(ee, row):
    sample = decode_sample(row)
    scenes = matched_scenes(ee, sample)
    values = dict(landsat_l2=np.zeros((4, 7, 120, 120), 'float32'),
                  landsat_l2_valid=np.zeros((4, 7, 120, 120), bool),
                  landsat_l2_time=np.zeros(4, 'int64'))
    images, names = [], []
    bands = [*LANDSAT_SR_BANDS, "QA_PIXEL", "QA_RADSAT", "valid"]
    for slot, scene in scenes.items():
        if scene is None:
            continue
        image = ee.Image(scene["asset_id"]).select(bands[:-1])
        valid = image.mask().reduce(ee.Reducer.min()).gt(0).rename("valid")
        renamed = [f"t{slot}_{band}" for band in bands]
        images.append(ee.Image.cat([image, valid]).unmask(0, sameFootprint=False)
                      .rename(renamed).toFloat())
        names.extend(renamed)
    if images:
        raw = compute_pixels(ee, ee.Image.cat(images), names, sample["crs"],
                             sample["transform"], 120, 120)
        for slot, scene in scenes.items():
            if scene is None:
                continue
            prefix = f"t{slot}_"
            sr, mask = landsat_l2_pixels(
                np.stack([raw[prefix + band] for band in LANDSAT_SR_BANDS]),
                raw[prefix + "QA_PIXEL"], raw[prefix + "QA_RADSAT"],
                raw[prefix + "valid"] > 0)
            index = int(slot)
            values['landsat_l2'][index] = sr
            values['landsat_l2_valid'][index] = mask
            values['landsat_l2_time'][index] = int(scene['time_ms'] // 1000)
    missing = [slot for slot in range(4) if sample['inputs'].get(f'landsat_{slot}')
               and not scenes.get(str(slot))]
    return values, dict(scenes=scenes, missing_l2_slots=missing,
                        policy="same_L1_acquisition;no_reselection;SR_only;nearest")
