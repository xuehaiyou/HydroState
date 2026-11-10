"""Station spatial operators; static ET never changes its fixed 210 m support."""

from pathlib import Path

import numpy as np
import zarr

from hydrostate.data.gee.backends.pixels import compute_pixels
from hydrostate.data.gee.contracts import fingerprint
from hydrostate.data.retrieval.catalog import cached_json
from hydrostate.data.retrieval.composites import DW_CLASSES, daily_collection

# IGBP metadata is independent of the model's predicted fractions.
IGBP_DW = {
    "ENF": {1},
    "EBF": {1},
    "DNF": {1},
    "DBF": {1},
    "MF": {1},
    "CSH": {5},
    "OSH": {5},
    "WSA": {1, 2, 5},
    "SAV": {1, 2, 5},
    "GRA": {2},
    "WET": {3},
    "CRO": {4},
    "WAT": {0},
    "SNO": {8},
    "BSV": {7},
    "URB": {6},
}


def static_quality(labels, probability, observed, source_cover, config):
    """Fractions always use all 441 ten-metre cells, including missing/unknown cells."""
    labels, probability, observed = map(np.asarray, (labels, probability, observed))
    if any(a.shape != (21, 21) for a in (labels, probability, observed)):
        raise ValueError("Expected exact 210 m static support on a 10 m grid")
    known = observed.astype(bool) & (probability >= config["class_confidence"])
    counts = np.array([(known & (labels == c)).sum() for c in range(9)])
    dominant = int(counts.argmax())
    fraction, valid = float(counts[dominant] / 441), float(observed.astype(bool).mean())
    expected = set().union(*(IGBP_DW.get(c, set()) for c in str(source_cover).split(";")))
    reasons = []
    if valid < config["spatial_valid_fraction"]:
        reasons.append("insufficient_contemporary_classification_coverage")
    if fraction < config["homogeneous_fraction"]:
        reasons.append("insufficient_homogeneous_whole_area")
    if not expected:
        reasons.append("unresolved_source_landcover")
    elif dominant not in expected:
        reasons.append("source_landcover_disagreement")
    return dict(
        accepted=not reasons,
        reasons=reasons,
        valid_fraction=valid,
        homogeneous_fraction=fraction,
        dominant_class=DW_CLASSES[dominant],
        expected_classes=sorted(expected),
        source_landcover=source_cover,
        operator="static_210m",
        approximation=True,
    )


def check_static(ee, sample, source_cover, cache, config):
    """Only a 21x21 classified patch is transferred before downloading model inputs."""
    transform = list(sample["transform"])
    transform[2] += 510
    transform[5] -= 510
    identity = dict(
        collection="GOOGLE/DYNAMICWORLD/V1",
        crs=sample["crs"],
        transform=transform,
        start=sample["target_time"],
        end=sample["target_end"],
        aggregation="daily_equal_probability_mean_v1",
    )

    def fetch():
        images = (
            ee.ImageCollection(identity["collection"])
            .filterBounds(ee.Geometry.Point([sample["lon"], sample["lat"]]))
            .filterDate(identity["start"], identity["end"])
            .select(DW_CLASSES)
        )
        daily = daily_collection(ee, images)
        empty = ee.Image.constant([0] * 9).rename(DW_CLASSES).toFloat().updateMask(ee.Image(0))
        mean = daily.merge(ee.ImageCollection([empty])).mean()
        observed = mean.mask().reduce(ee.Reducer.min()).gt(0).rename("observed")
        label = mean.toArray().arrayArgmax().arrayGet([0]).rename("label")
        confidence = mean.reduce(ee.Reducer.max()).rename("confidence")
        image = ee.Image.cat(
            [label.unmask(-1, False), confidence.unmask(0, False), observed.unmask(0, False)]
        ).toFloat()
        raw = compute_pixels(
            ee, image, ["label", "confidence", "observed"], sample["crs"], transform, 21, 21
        )
        return dict(identity=identity, **{k: raw[k].tolist() for k in raw.dtype.names})

    name = f"{fingerprint(identity)}.json"
    path = Path(cache) / "static" / name
    # Read through immutable prior caches; changed thresholds are evaluated below.
    if not path.exists():
        for prior in config.get("reuse_cache_roots", []):
            candidate = Path(prior) / "static" / name
            if candidate.exists():
                path = candidate
                break
    result = cached_json(path, fetch)
    quality = static_quality(
        result["label"], result["confidence"], result["observed"], source_cover, config
    )
    quality["query_identity"] = identity
    return quality


def static_weights():
    weights = np.zeros((40, 40), dtype="float32")
    weights[17:24, 17:24] = 1 / 49
    return weights


def footprint_operator(path, input_valid, minimum_mass):
    """Retain absolute captured mass; only normalize after the effective-domain gate."""
    group = zarr.open_group(str(path), mode="r")
    if not group.attrs.get("complete") or not group.attrs.get("footprint_tile_valid"):
        raise ValueError("Footprint cache is not accepted/complete")
    mass = group["mass"][:]
    if mass.shape != (40, 40) or not np.isfinite(mass).all() or (mass < 0).any():
        raise ValueError("Invalid footprint mass")
    kept = np.where(input_valid, mass, 0)
    captured = float(kept.sum())
    quality = dict(
        operator="footprint",
        tile_mass=float(mass.sum()),
        effective_mass=captured,
        accepted=captured >= minimum_mass,
    )
    weights = kept / captured if quality["accepted"] else np.zeros_like(kept)
    return weights.astype("float32"), quality
