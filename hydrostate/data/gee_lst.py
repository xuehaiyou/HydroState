"""One quality-screened LST observation per existing sample, at its actual time."""

from datetime import UTC, datetime

import numpy as np

from .gee_labels import COLLECTIONS, _metadata, _tagged, convert_lst_pixels, decode_sample
from .target_times import epoch_seconds, scene_time_bounds

PRODUCER_VERSION = "gee_lst_observation_v1"
MAX_UNCERTAINTY_K = 2.0


def variables():
    return dict(lst=dict(shape=[1, 40, 40], grid="output", units="K",
                         effective_resolution_m=100, uncertainty_units="K",
                         default_max_uncertainty=MAX_UNCERTAINTY_K, time_support="instant"))


def input_scene_pairs(sample):
    """Match satellite/path/row/acquisition identity, not merely a nearby date."""
    pairs = {}
    for key, scene in sample["inputs"].items():
        if not key.startswith("landsat_") or not scene:
            continue
        asset = scene["asset_id"]
        prefix, identity = asset.rsplit("/", 1)
        if prefix not in {"LANDSAT/LC08/C02/T1", "LANDSAT/LC09/C02/T1"}:
            raise ValueError(f"Unsupported Landsat input collection: {prefix}")
        if identity.split("_")[0] != prefix.split("/")[1]:
            raise ValueError("Landsat satellite identity disagrees with its collection")
        target = prefix + "_L2/" + identity
        pairs[target] = dict(input_asset=asset, input_slot=int(key.rsplit("_", 1)[1]),
                             input_time_ms=scene["system:time_start"])
    return pairs


def order_candidates(candidates, sample):
    """All same-day candidates first, then closest paired input; stable tie breaks."""
    target = epoch_seconds(sample["target_time"]) * 1000
    day = sample["target_time"]
    unique = {}
    for raw in candidates:
        scene = dict(raw)
        scene["date"] = str(datetime.fromtimestamp(scene["time_ms"] / 1000, UTC).date())
        if scene["date"] != day and "input_slot" not in scene:
            raise ValueError("Off-date LST must be paired to an existing Landsat input")
        if "input_time_ms" in scene and scene["time_ms"] != scene["input_time_ms"]:
            raise ValueError("Paired Landsat L1/L2 acquisition times disagree")
        unique[scene["asset_id"]] = scene
    return sorted(unique.values(), key=lambda s: (
        s["date"] != day, abs(s["time_ms"] - target), s["asset_id"]))


def select_candidates(ee, sample):
    pairs = input_scene_pairs(sample)
    tr = sample["transform"]
    region = ee.Geometry.Rectangle([tr[2], tr[5] - 1200, tr[2] + 1200, tr[5]],
                                   sample["crs"], False)
    all_lst = _tagged(ee, COLLECTIONS["lst8"]).merge(_tagged(ee, COLLECTIONS["lst9"]))
    all_lst = all_lst.filterBounds(region).filter(ee.Filter.eq("PROCESSING_LEVEL", "L2SP"))
    start = ee.Date(sample["target_time"])
    same_day = all_lst.filterDate(start, start.advance(1, "day"))
    paired = all_lst.filter(ee.Filter.inList("source_asset", list(pairs)))
    candidates = same_day.merge(paired).distinct("source_asset")
    # Earth Engine rejects toList(0), even for a legitimately empty collection.
    metadata = candidates.toList(candidates.size().max(1)).map(
        lambda value: _metadata(ee, ee.Image(value), "ST_B10"))
    result = metadata.getInfo()
    for scene in result:
        scene.update(pairs.get(scene["asset_id"], {}))
    return order_candidates(result, sample)


def fetch_pixels(ee, sample, scene):
    image = ee.Image(scene["asset_id"]).select(
        ["ST_B10", "QA_PIXEL", "ST_QA"], ["lst_raw", "lst_qa", "lst_uncertainty_raw"])
    valid = image.mask().reduce(ee.Reducer.min()).unmask(0, sameFootprint=False)
    image = image.unmask(0, sameFootprint=False).addBands(valid.rename("lst_valid")).toFloat()
    names = ["lst_raw", "lst_qa", "lst_uncertainty_raw", "lst_valid"]
    raw = ee.data.computePixels(dict(
        expression=image, fileFormat="NUMPY_NDARRAY", bandIds=names,
        grid=dict(dimensions=dict(width=120, height=120), crsCode=sample["crs"],
                  affineTransform=dict(zip(
                      ("scaleX", "shearX", "translateX", "shearY", "scaleY", "translateY"),
                      sample["transform"])))))
    if raw.shape != (120, 120) or list(raw.dtype.names or []) != names:
        raise ValueError("Unexpected LST pixel response")
    data = {name: raw[name] for name in names}
    if any(not np.isfinite(value).all() for value in data.values()):
        raise ValueError("Nonfinite LST pixel response")
    return convert_lst_pixels(data)


def eligible(values, observed):
    return (values["lst_valid"] & np.isfinite(values["lst_uncertainty"])
            & (values["lst_uncertainty"] >= 0)
            & (values["lst_uncertainty"] <= MAX_UNCERTAINTY_K) & observed)


def choose_observation(sample, observed, paired_observed, previous, candidates, fetch):
    """Test actual window pixels before selecting a scene; errors must propagate.

    ``observed`` is the existing model input support at 30 m. Paired observations
    additionally require valid pixels in the corresponding Landsat input image.
    Candidate metadata and pixel downloads are lazy, so preserving a baseline
    label never requires a GEE request.
    """
    trials = []

    def accept(values, scene, method, support):
        mask = eligible(values, support)
        trials.append(dict(asset_id=scene["asset_id"], method=method,
                           clear_pixels=int(values["lst_valid"].sum()),
                           eligible_pixels=int(mask.sum())))
        if not mask.any():
            return None
        result = dict(lst=values["lst"], lst_valid=mask,
                      lst_uncertainty=values["lst_uncertainty"],
                      lst_time=scene_time_bounds(scene))
        return result, dict(scenes=dict(lst=scene), selection=method, trials=trials)

    if previous is not None:
        values, scene = previous
        if scene and str(datetime.fromtimestamp(scene["time_ms"] / 1000, UTC).date()) == (
                sample["target_time"]):
            kept = accept(values, scene, "preserved_same_day", observed)
            if kept is not None:
                return kept
    attempted = {item["asset_id"] for item in trials}
    for scene in order_candidates(candidates(), sample):
        if scene["asset_id"] in attempted:
            continue
        same_day = scene["date"] == sample["target_time"]
        support = observed if same_day else observed & paired_observed[scene["input_slot"]]
        selected = accept(fetch(scene), scene, "same_day" if same_day else "paired_input", support)
        if selected is not None:
            return selected
    shape = (1, 40, 40)
    return (dict(lst=np.zeros(shape, "float32"), lst_valid=np.zeros(shape, bool),
                 lst_uncertainty=np.zeros(shape, "float32"), lst_time=np.zeros(2, "int64")),
            dict(scenes=dict(lst=None), selection="missing", trials=trials))


def fetch_sample(ee, row, observed, paired_observed, previous=None):
    sample = decode_sample(row)
    return choose_observation(sample, observed, paired_observed, previous,
                              lambda: select_candidates(ee, sample),
                              lambda scene: fetch_pixels(ee, sample, scene))
