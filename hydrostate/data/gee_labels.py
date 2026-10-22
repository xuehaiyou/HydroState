"""GEE product adapters for extending an existing immutable sample manifest."""

from __future__ import annotations

import json
import time
from datetime import date, timedelta

import numpy as np

from hydrostate.data.native_support import NativeSupport

PRODUCER_VERSION = "gee_ecosystem_labels_v2"
COLLECTIONS = dict(
    dw="GOOGLE/DYNAMICWORLD/V1", lai="MODIS/061/MCD15A3H",
    gpp="projects/pml_evapotranspiration/PML/OUTPUT/PML_V22a",
    albedo="MODIS/061/MCD43A3",
    lst8="LANDSAT/LC08/C02/T1_L2", lst9="LANDSAT/LC09/C02/T1_L2",
)
COVER_NAMES = ("water", "trees", "grass", "flooded_vegetation", "crops",
               "shrub_and_scrub", "built", "bare", "snow_and_ice")
NATIVE_NAMES = ("lai", "gpp", "albedo_bsa", "albedo_wsa")


def variables(capacity=64):
    specs = {name: dict(shape=[1, 120, 120], grid="input", units="1")
             for name in COVER_NAMES}
    for name in NATIVE_NAMES:
        units = {"lai": "m2/m2", "gpp": "gC/m2/day"}.get(name, "1")
        specs[name] = dict(shape=[capacity], spatial_shape=[40, 40], grid="native", units=units)
    specs["lst"] = dict(shape=[1, 40, 40], grid="output", units="K",
                        effective_resolution_m=100, uncertainty_units="K",
                        default_max_uncertainty=2.0)
    return specs


def tasks():
    from hydrostate.tasks import LEGACY_TASKS, task_config

    return (task_config(LEGACY_TASKS)
            | {name: dict(support="pixel", grid="input") for name in COVER_NAMES[1:]}
            | {name: dict(support="native", grid="native") for name in NATIVE_NAMES}
            | {"lst": dict(support="pixel", grid="output")})


def decode_sample(row):
    sample = dict(row)
    for key in ("labels", "transform", "inputs"):
        if isinstance(sample.get(key), str):
            sample[key] = json.loads(sample[key])
    sample["target_time"] = str(sample["target_time"])[:10]
    if (sample.get("height", 120) != 120 or sample.get("width", 120) != 120
            or not np.isclose(abs(sample["transform"][0]), 10)
            or not np.isclose(abs(sample["transform"][4]), 10)):
        raise ValueError("This producer requires the existing 120 x 120, 10 m sample grid")
    return sample


def containing_interval(start, days):
    start = date.fromisoformat(start[:10])
    return str(start), str(min(start + timedelta(days=days), date(start.year + 1, 1, 1)))


def _metadata(ee, image, band):
    projection = image.select(band).projection()
    return ee.Dictionary(dict(asset_id=image.get("source_asset"),
                              time_ms=image.get("system:time_start"),
                              projection=projection))


def _tagged(ee, collection):
    return ee.ImageCollection(collection).map(
        lambda im: im.set("source_asset", ee.String(collection + "/").cat(im.get("system:index"))))


def select_scenes(ee, sample):
    """One metadata request for all products; absence is distinct from request errors."""
    target = date.fromisoformat(sample["target_time"])
    selections = {}
    dw = sample["labels"]["dw"]
    if not dw:
        raise ValueError("Existing DW asset is required for consistent coverage fractions")
    image = ee.Image(dw["asset_id"]).set("source_asset", dw["asset_id"])
    selections["dw"] = _metadata(ee, image, "label")
    for product, band, days in (("lai", "Lai", 4), ("gpp", "GPP", 8),
                                ("albedo", "Albedo_BSA_shortwave", 1)):
        existing = sample["labels"].get("pml") if product == "gpp" else None
        if existing:
            if not existing["asset_id"].startswith(COLLECTIONS["gpp"] + "/"):
                raise ValueError("Existing ET uses another PML version; cannot mix products")
            image = ee.Image(existing["asset_id"]).set("source_asset", existing["asset_id"])
            selections[product] = _metadata(ee, image, band)
        else:
            collection = _tagged(ee, COLLECTIONS[product]).filterDate(
                str(target - timedelta(days=days - 1)), str(target + timedelta(days=1)))
            image = ee.Image(collection.sort("system:time_start", False).first())
            selections[product] = ee.Algorithms.If(
                collection.size().gt(0), _metadata(ee, image, band), None)
    tr = sample["transform"]
    region = ee.Geometry.Rectangle([tr[2], tr[5] - 1200, tr[2] + 1200, tr[5]],
                                   sample["crs"], False)
    landsat = _tagged(ee, COLLECTIONS["lst8"]).merge(_tagged(ee, COLLECTIONS["lst9"]))
    landsat = (landsat.filterBounds(region)
               .filterDate(str(target), str(target + timedelta(days=1)))
               .filter(ee.Filter.eq("PROCESSING_LEVEL", "L2SP")))
    image = ee.Image(landsat.sort("CLOUD_COVER").first())
    selections["lst"] = ee.Algorithms.If(
        landsat.size().gt(0), _metadata(ee, image, "ST_B10").set(
            "same_day_candidates", landsat.size()), None)
    result = ee.Dictionary(selections).getInfo()
    for product in ("dw", "lai", "gpp", "albedo", "lst"):
        scene = result.setdefault(product, None)
        if scene is None:
            continue
        from datetime import UTC, datetime

        scene_date = datetime.fromtimestamp(scene["time_ms"] / 1000, UTC).date()
        scene["date"] = str(scene_date)
        if product in {"lai", "gpp"}:
            start, end = containing_interval(str(scene_date), 4 if product == "lai" else 8)
            if not start <= str(target) < end:
                raise ValueError(f"{product} composite does not contain target date")
            scene.update(interval_start=start, interval_end_exclusive=end)
        elif product == "albedo":
            # MCD43's nominal date is day 9 of its 16-day retrieval window.
            scene.update(interval_start=str(scene_date - timedelta(days=8)),
                         interval_end_exclusive=str(scene_date + timedelta(days=8)))
        elif product in {"dw", "lst"} and scene_date != target:
            raise ValueError(f"{product} scene is not on the target UTC date")
    return result


def build_image(ee, scenes):
    """Fetch raw quality fields for explicit, testable masks before publication."""
    parts, names = [], []
    sources = dict(
        dw={"label": "dw_class"},
        lai={"Lai": "lai_raw", "FparLai_QC": "lai_qc", "FparExtra_QC": "lai_extra"},
        gpp={"GPP": "gpp_raw"},
        albedo={"Albedo_BSA_shortwave": "albedo_bsa_raw",
                "Albedo_WSA_shortwave": "albedo_wsa_raw",
                "BRDF_Albedo_Band_Mandatory_Quality_shortwave": "albedo_qc"},
        lst={"ST_B10": "lst_raw", "QA_PIXEL": "lst_qa", "ST_QA": "lst_uncertainty_raw"},
    )
    for product, mapping in sources.items():
        scene = scenes[product]
        image = (ee.Image(scene["asset_id"]).select(list(mapping), list(mapping.values()))
                 if scene else ee.Image.constant([0] * len(mapping)).rename(list(mapping.values()))
                 .updateMask(ee.Image(0)))
        valid = image.mask().reduce(ee.Reducer.min()).unmask(0, sameFootprint=False)
        parts.extend((image.unmask(0, sameFootprint=False), valid.rename(product + "_valid")))
        names.extend([*mapping.values(), product + "_valid"])
    return ee.Image.cat(parts).toFloat(), names


def quality_masks(data, lst_max_uncertainty_k=float("inf")):
    """Physical ranges and documented QA bits, identical for online and offline checks."""
    dw = (data["dw_valid"] > 0) & np.isin(data["dw_class"], np.arange(9))
    qc, extra = data["lai_qc"].astype("uint16"), data["lai_extra"].astype("uint16")
    lai = ((data["lai_valid"] > 0) & ((qc & 1) == 0) & (((qc >> 3) & 3) == 0)
           & ((qc >> 5) <= 1) & ((extra & 124) == 0)
           & (data["lai_raw"] >= 0) & (data["lai_raw"] <= 100))
    gpp = ((data["gpp_valid"] > 0) & (data["gpp_raw"] >= 0)
           & (data["gpp_raw"] <= 10000))
    albedo = (data["albedo_valid"] > 0) & (data["albedo_qc"] == 0)
    for name in ("albedo_bsa_raw", "albedo_wsa_raw"):
        albedo &= (data[name] >= 0) & (data[name] <= 1000)
    qa = data["lst_qa"].astype("uint16")
    # Bits 0..5: fill, dilated cloud, cirrus, cloud, shadow, snow. Snow is kept:
    # unlike cloudy retrievals, a clear snow-surface temperature remains meaningful.
    lst = ((data["lst_valid"] > 0) & ((qa & 31) == 0)
           & (data["lst_raw"] > 0) & (data["lst_raw"] <= 65535)
           & (data["lst_uncertainty_raw"] * .01 <= lst_max_uncertainty_k))
    return dict(dw=dw, lai=lai, gpp=gpp, albedo=albedo, lst=lst)


def convert_lst_pixels(data):
    """Conservative 10 m sampling -> 30 m label support, with correlated errors."""
    qa = data["lst_qa"].astype("uint16")
    clear = ((data["lst_valid"] > 0) & ((qa & 31) == 0)
             & (data["lst_raw"] > 0) & (data["lst_raw"] <= 65535))
    valid = clear.reshape(40, 3, 40, 3).all(axis=(1, 3))
    temperature = (data["lst_raw"] * .00341802 + 149).reshape(40, 3, 40, 3).mean(axis=(1, 3))
    uncertainty = (data["lst_uncertainty_raw"] * .01).reshape(40, 3, 40, 3).max(axis=(1, 3))
    return dict(lst=np.where(valid, temperature, 0)[None], lst_valid=valid[None],
                lst_uncertainty=np.where(valid, uncertainty, 0)[None])


def convert_pixels(sample, scenes, data, *, capacity=64):
    masks = quality_masks(data)
    values = {}
    for code, name in enumerate(COVER_NAMES):
        values[name] = ((data["dw_class"] == code) & masks["dw"])[None].astype("float32")
        values[name + "_valid"] = masks["dw"][None]
    native = NativeSupport(sample)
    scalings = dict(lai=.1, gpp=.01, albedo_bsa=.001, albedo_wsa=.001)
    for name, scale in scalings.items():
        source = "albedo" if name.startswith("albedo_") else name
        scene = scenes[source]
        targets = native.targets(data[name + "_raw"] * scale, masks[source],
                                 scene["projection"] if scene else {}, capacity)
        values.update({name: targets["value"], name + "_valid": targets["valid"],
                       name + "_weights": targets["weights"]})
    values.update(convert_lst_pixels(data))
    return values


def fetch_sample(ee, row, *, capacity=64):
    started = time.perf_counter()
    sample = decode_sample(row)
    scenes = select_scenes(ee, sample)
    selected = time.perf_counter()
    image, names = build_image(ee, scenes)
    from hydrostate.data.gee.backends.pixels import compute_pixels

    raw = compute_pixels(ee, image, names, sample["crs"], sample["transform"], 120, 120)
    data = {name: raw[name] for name in names}
    if any(not np.isfinite(value).all() for value in data.values()):
        raise ValueError("Nonfinite GEE response")
    downloaded = time.perf_counter()
    values = convert_pixels(sample, scenes, data, capacity=capacity)
    qa = quality_masks(data)
    return values, dict(scenes=scenes, valid_input_pixels={k: int(v.sum()) for k, v in qa.items()},
                        selection_s=selected - started,
                        compute_receive_s=downloaded - selected,
                        conversion_s=time.perf_counter() - downloaded,
                        decoded_bytes=raw.nbytes)
