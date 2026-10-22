"""Matched-window GEE audit of three coverage confidence policies."""

import argparse
import json
from pathlib import Path

from hydrostate.data.gee.client import initialize
from hydrostate.data.spatial import grid
from hydrostate.data.retrieval.collector import catalog
from hydrostate.data.retrieval.composites import (
    DW_CLASSES,
    cover_composite,
    mean_composite,
    prepare_cover,
)
from hydrostate.data.retrieval.gee_labels import pixels
from hydrostate.data.retrieval.labels import cover_targets

THRESHOLDS = dict(water=0.35, snow_ice=0.60, land=0.85, land_max_water_ice=0.10)


def former_cover(ee, image, relaxed=False):
    probabilities = image.select(DW_CLASSES)
    winner = probabilities.toArray().arrayArgmax().arrayGet([0])
    water, ice = winner.eq(0), winner.eq(8)
    land = water.Or(ice).Not()
    known = (
        water.And(image.select("water").gt(0.35))
        .Or(ice.And(image.select("snow_and_ice").gt(0.35)))
        .Or(land)
        if relaxed
        else probabilities.reduce(ee.Reducer.max()).gte(0.8)
    )
    return ee.Image(
        ee.Image.cat(
            [
                water.rename("water"),
                ice.rename("snow_ice"),
                land.rename("land"),
            ]
        )
        .updateMask(known)
        .addBands(known.Not().rename("unknown"))
        .toFloat()
        .copyProperties(image, ["system:time_start"])
    )


def check_rules(ee):
    cases = [
        ("water_at_threshold", [0.35, 0.3, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05], [1, 0, 0]),
        (
            "water_above_threshold",
            [0.36, 0.29, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
            [1, 0, 0],
        ),
        ("ice_below_threshold", [0.05, 0.29, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.36], [0, 0, 0]),
        ("ice_at_threshold", [0.05, 0.35, 0, 0, 0, 0, 0, 0, 0.60], [0, 1, 0]),
        ("water_not_top1", [0.36, 0.4, 0.04, 0.04, 0.04, 0.03, 0.03, 0.03, 0.03], [0, 0, 0]),
        ("land_probability_sum", [0.07, 0.2, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.07], [0, 0, 1]),
        ("land_at_threshold", [0.09, 0.85, 0, 0, 0, 0, 0, 0, 0.06], [0, 0, 1]),
        ("water_at_land_exclusion", [0.1, 0.9, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0]),
        ("ice_at_land_exclusion", [0, 0.9, 0, 0, 0, 0, 0, 0, 0.1], [0, 0, 0]),
        ("land_below_threshold", [0.08, 0.84, 0, 0, 0, 0, 0, 0, 0.08], [0, 0, 0]),
    ]
    features = []
    for name, values, expected in cases:
        image = ee.Image.constant(values).rename(DW_CLASSES).set("system:time_start", 0)
        result = prepare_cover(ee, image, THRESHOLDS).unmask(0)
        value = result.reduceRegion(ee.Reducer.first(), ee.Geometry.Point([0, 0]), scale=30)
        features.append(ee.Feature(None, value).set("case", name))
    observed = ee.FeatureCollection(features).getInfo()["features"]
    for feature, (name, _, expected) in zip(observed, cases):
        assert [feature["properties"][k] for k in ("water", "snow_ice", "land")] == expected, name
        assert feature["properties"]["unknown"] == (not any(expected)), name
    return [name for name, _, _ in cases]


def audit(project, output, cache):
    ee = initialize(project)
    checks = check_rules(ee)
    rows = []
    start, end = "2020-07-11", "2020-07-19"
    for name, lon, lat, block in [
        ("Lake Tahoe", -120.05, 39.1, [-122, 38, 2]),
        ("Greenland", -45.0, 70.0, [-46, 70, 2]),
        ("Ames", -93.63, 41.99, [-94, 40, 2]),
        ("Forest", -120.40333727688727, 38.792835449205825, [-122, 38, 2]),
        ("Great Plains", -100.0, 35.0, [-100, 34, 2]),
    ]:
        crs, transform = grid(lon, lat)
        sample = dict(lon=lon, lat=lat, crs=crs, transform=transform, block=block)
        scenes = catalog(ee, cache, "dw", sample, start, end)
        ids = [s["asset_id"] for s in scenes]
        variants = []
        for relaxed in (False, True):
            former = ee.ImageCollection.fromImages([ee.Image(sid) for sid in ids]).map(
                lambda image: former_cover(ee, image, relaxed=relaxed)
            )
            old = mean_composite(ee, former, ["water", "snow_ice", "land"], start)
            old = old.addBands(old.select("days").gt(0).rename("known"))
            quality = mean_composite(ee, former, ["unknown"], start).select(
                ["unknown", "days"], ["unknown_fraction", "observed_days"]
            )
            variants.append(("relaxed_0.35" if relaxed else "former_0.8", old.addBands(quality)))
        variants.append(("current_0.35_0.60_0.85", cover_composite(ee, ids, start, THRESHOLDS)))
        record = dict(name=name, lon=lon, lat=lat, scenes=len(ids))
        for label, image in variants:
            if ids:
                image = image.setDefaultProjection(ee.Image(ids[0]).select("label").projection())
            raw = pixels(
                ee,
                image,
                [
                    "water",
                    "snow_ice",
                    "land",
                    "known",
                    "days",
                    "first_day",
                    "last_day",
                    "unknown_fraction",
                    "observed_days",
                ],
                sample,
            )
            targets = cover_targets({k: raw[k] for k in raw.dtype.names})
            valid = targets["water_valid"] > 0
            record[label] = dict(
                valid_pixels_30m=int(valid.sum()),
                total_pixels_30m=1600,
                known_area_fraction=float(raw["known"].mean()),
                observed_area_fraction=float((raw["observed_days"] > 0).mean()),
                mean_unknown_fraction=float(
                    raw["unknown_fraction"][raw["observed_days"] > 0].mean()
                )
                if (raw["observed_days"] > 0).any()
                else None,
                **{
                    f"pure_{cls}_pixels": int((valid & (targets[cls] > 0.95)).sum())
                    for cls in ("water", "snow_ice", "land")
                },
            )
        rows.append(record)
        print(json.dumps(record), flush=True)
    report = dict(
        start=start,
        end=end,
        rule_checks=checks,
        thresholds=THRESHOLDS,
        samples=rows,
        limitation="Five illustrative windows; availability, not accuracy or global yield.",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default="cropland-477906")
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/retrieval/cover_threshold_audit.json")
    )
    parser.add_argument("--cache", type=Path, default=Path("outputs/retrieval/audit_cache"))
    args = parser.parse_args()
    audit(args.project, args.output, args.cache)
