"""Immutable validation pairing of native products and station observation operators."""

import argparse
import json
import runpy
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from affine import Affine
from pyproj import Transformer

from hydrostate.data.gee.client import initialize, retry
from hydrostate.data.gee.contracts import atomic_json, fingerprint, map_point
from hydrostate.data.gee.products import collection
from hydrostate.data.retrieval.catalog import cached_json
from hydrostate.data.retrieval.composites import cover_composite
from hydrostate.data.retrieval.specs import LABEL_SPECS, SOURCES
from hydrostate.data.site.archives import sha256
from hydrostate.data.site.comparison import LocalSMAP, pair_sm, smap_observations
from hydrostate.data.site.et_comparison import LocalPML, period_end, pml_observation

VERSION = "station_product_validation_v1"


@lru_cache(maxsize=16)
def observation_metadata(path):
    group = zarr.open_group(path, mode="r")
    if "observation_records" in group.attrs:
        return group.attrs["observation_records"]
    return None


def observations(release, split="validation"):
    """Stable observation keys preserve co-located sensors and packed Zarr row identity."""
    release = Path(release)
    definitions = json.loads((release / "datasets.json").read_text())["datasets"]
    for cohort, spec in enumerate(definitions):
        root = (release / spec["data_root"]).resolve()
        table = pd.read_parquet(release / spec["manifest"])
        for row in table[table.split.eq(split)].to_dict("records"):
            path = root / row["label_path"]
            labels = zarr.open_group(str(path), mode="r")
            records = observation_metadata(str(path))
            if records is None:
                packed = zarr.open_group(str(path.parent), mode="r")
                records = json.loads(str(packed["observation_metadata"][row["shard_index"]]))
            for task in ("sm", "et"):
                valid = labels[f"site_{task}_native_valid"][row["shard_index"]].astype(bool)
                for index in np.flatnonzero(valid):
                    record = records[index]["label"]
                    target = float(labels[f"site_{task}_native_value"][row["shard_index"], index])
                    if not np.isclose(target, record[task]["value"], rtol=1e-5, atol=1e-7):
                        raise ValueError("Station value differs from observation metadata")
                    yield dict(
                        key=f"{cohort}:{row['sample_id']}:{task}:{index}", cohort=cohort,
                        sample_id=row["sample_id"], observation_index=int(index), task=task,
                        spatial_group_id=row["spatial_group_id"], split=split, record=record,
                        label_path=str(path), shard_index=int(row["shard_index"]),
                        crs=row["crs"], transform=json.loads(row["transform"]),
                        original_target=target, frequency_bin=int(row["frequency_bin"]),
                        et_support=row["et_support"],
                    )


def native_operator(item, projection):
    """Apply the same 30 m station operator to nearest native PML cells, without interpolation."""
    labels = zarr.open_group(item["label_path"], mode="r")
    weights = labels["site_et_native_weights"][item["shard_index"], item["observation_index"]]
    if not np.isfinite(weights).all() or (weights < 0).any() or not np.isclose(weights.sum(), 1):
        raise ValueError("Invalid station ET spatial weights")
    y, x = np.where(weights > 0)
    px, py = map_point(Affine(*item["transform"]), (x + .5) * 3, (y + .5) * 3)
    nx, ny = Transformer.from_crs(item["crs"], projection["crs"], always_xy=True).transform(px, py)
    col, row = np.floor(map_point(~Affine(*projection["transform"]), nx, ny)).astype(int)
    cells = {}
    for c, r, weight in zip(col, row, weights[y, x]):
        cells[(int(c), int(r))] = cells.get((int(c), int(r)), 0.) + float(weight)
    return [dict(col=c, row=r, weight=w) for (c, r), w in sorted(cells.items())]


def reduce_pml_cells(cells, min_land_fraction):
    """Never renormalize over missing/impure source cells or fill missing ET with zero."""
    if not cells or not np.isclose(sum(c["weight"] for c in cells), 1, atol=1e-5):
        raise ValueError("PML operator weights must sum to one")
    if any(c.get("land_purity") is None or c["land_purity"] <= min_land_fraction for c in cells):
        return None, "PML_native_land_purity_not_above_95_percent"
    if any(c.get("ET") is None or not np.isfinite(c["ET"]) or c["ET"] < 0 for c in cells):
        return None, "PML_missing_or_invalid_support"
    return sum(c["weight"] * c["ET"] for c in cells), None


def pml_support(ee, item, cache, local, quality):
    record = item["record"]
    point = pml_observation(ee, record, cache / "pml_points", local_products=local)
    if not point.get("projection"):
        return dict(value=None, reason=point.get("reason", "PML_missing"), point=point)
    cells = native_operator(item, point["projection"])
    identity = dict(version=VERSION, point=point["identity"], cells=cells,
                    quality=quality)

    def fetch():
        projection = point["projection"]
        grid = Affine(*projection["transform"])
        shapes, centres = [], []
        for i, cell in enumerate(cells):
            c, r = cell["col"], cell["row"]
            corners = [list(map_point(grid, x, y)) for x, y in
                       ((c, r), (c + 1, r), (c + 1, r + 1), (c, r + 1), (c, r))]
            shape = ee.Geometry.Polygon([corners], proj=projection["crs"], geodesic=False)
            shapes.append(ee.Feature(shape, dict(index=i)))
            centres.append(ee.Feature(ee.Geometry.Point(list(map_point(grid, c + .5, r + .5)),
                                                       proj=projection["crs"]), dict(index=i)))
        polygons = ee.FeatureCollection(shapes)
        dw = collection(ee, "dw").filterBounds(polygons.geometry()).filterDate(
            record["target_time"], record["target_end"])
        ids = dw.aggregate_array("asset_id").getInfo()
        if not ids:
            return dict(value=None, reason="no_DW_for_PML_native_purity", cells=cells)
        cover = cover_composite(ee, ids, record["target_time"], quality["thresholds"])
        # Unknown/unobserved area remains in the denominator, as in product supervision.
        purity = cover.select("land").unmask(0, False).reduceRegions(
            collection=polygons, reducer=ee.Reducer.mean(), scale=10,
            crs=item["crs"], tileScale=2).getInfo()["features"]
        image = collection(ee, "pml").filterDate(
            record["target_time"], record["target_end"]).first()
        values = image.select("ET").multiply(LABEL_SPECS["et_land"].scale).reduceRegions(
            collection=ee.FeatureCollection(centres), reducer=ee.Reducer.first(),
            crs=projection["crs"], crsTransform=projection["transform"]).getInfo()["features"]
        land = {p["properties"]["index"]: p["properties"].get("mean") for p in purity}
        et = {p["properties"]["index"]: p["properties"].get("first") for p in values}
        result = [dict(c, ET=et.get(i), land_purity=land.get(i)) for i, c in enumerate(cells)]
        value, reason = reduce_pml_cells(result, quality["min_land_fraction"])
        return dict(value=value, reason=reason, cells=result, asset_id=point.get("asset_id"),
                    spatial_support="station_30m_operator_on_native_cells; centre_quadrature",
                    time_support="PML_UTC_period_vs_station_local_days; offset_recorded")

    return cached_json(cache / "pml_support" / f"{fingerprint(identity)}.json", fetch)


def product_row(ee, item, cache, quality, sm_local=None, et_local=None, reuse_sm=()):
    record = item["record"]
    result = {k: v for k, v in item.items() if k not in ("record", "transform")}
    result.update(target_time=record["target_time"], target_end=record["target_end"],
                  site_id=record["site_id"], group_id=record["group_id"],
                  product_value=None, station_value=None, reason=None)
    if item["task"] == "sm":
        if record["target_end"] <= min(s.start for s in SOURCES["sm"] if s.start):
            return dict(result, reason="before_SMAP_availability")
        product = smap_observations(ee, record, cache / "smap", reuse_caches=reuse_sm,
                                    local_products=sm_local)
        paired = pair_sm(record, product)
        result.update(product_value=paired["product_value"], station_value=paired["site_value"],
                      reason=paired["reason"], evidence=paired,
                      time_support="common_QA_morning_dates; model_is_8day_estimate")
    else:
        if record["target_end"] != period_end(record["target_time"]):
            raise ValueError("ET window is not a nominal PML period")
        product = pml_support(ee, item, cache, et_local, quality)
        result.update(product_value=product["value"], station_value=item["original_target"],
                      reason=product["reason"], evidence=product,
                      station_time_basis=record.get("time_basis"),
                      utc_offset_hours=record.get("utc_offset_hours"))
    result["eligible"] = result["reason"] is None
    return result


def collect(release, output, project, *, workers=4, limit=None, product_manifest=None,
            collection_config="configs/data/collection/retrieval.py"):
    """Resume per-observation results; transport errors remain retryable, never missing labels."""
    if not 1 <= workers <= 4:
        raise ValueError("Use 1–4 bounded workers")
    release, output = Path(release).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = runpy.run_path(str(collection_config))
    quality = dict(min_land_fraction=config["min_known_fraction"], thresholds=dict(
        water=config["water_confidence"], snow_ice=config["ice_confidence"],
        land=config["land_confidence"],
        land_max_water_ice=config["land_max_water_ice_probability"]))
    identity = dict(version=VERSION, release=str(release), split="validation", quality=quality,
                    inputs={p.name: sha256(p) for p in [release / "datasets.json",
                            *sorted(release.glob("cohort_*.parquet"))]},
                    sources={k: [vars(s) for s in SOURCES[k]] for k in ("sm", "pml", "dw")})
    plan = output / "plan.json"
    if plan.exists() and json.loads(plan.read_text()) != identity:
        raise ValueError("Baseline plan changed; use a new output directory")
    atomic_json(plan, identity)
    ee = initialize(project)
    sm_local = LocalSMAP(product_manifest) if product_manifest else None
    et_local = LocalPML(product_manifest) if product_manifest else None
    items = list(observations(release))
    if limit is not None:
        # Small representative pilot, not a performance/accuracy sample.
        chosen = []
        for task in ("sm", "et"):
            chosen.extend([r for r in items if r["task"] == task
                           and "2017" <= r["record"]["target_time"] < "2025"][:limit])
        items = chosen
    rows, failures = [], []
    (output / "rows").mkdir(exist_ok=True)

    def query(item):
        path = output / "rows" / f"{fingerprint(item['key'])}.json"
        if path.exists():
            row = json.loads(path.read_text())
            if row["key"] != item["key"]:
                raise ValueError("Baseline row identity mismatch")
            return row
        try:
            row = retry(
                lambda: product_row(ee, item, output / "cache", quality, sm_local, et_local),
                attempts=3)
        except Exception as error:
            return dict(key=item["key"], task=item["task"], error=repr(error))
        atomic_json(path, row)
        return row

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for row in pool.map(query, items):
            (failures if "error" in row else rows).append(row)
            if (len(rows) + len(failures)) % 20 == 0:
                atomic_json(output / "progress.json", dict(done=len(rows), errors=len(failures),
                            total=len(items), eligible=sum(r.get("eligible", False) for r in rows)))
    with (output / "pairs.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    atomic_json(output / "errors.json", failures)
    result = dict(status="completed" if not failures else "incomplete", total=len(items),
                  downloaded=len(rows), errors=len(failures),
                  eligible=dict(Counter(r["task"] for r in rows if r.get("eligible"))),
                  excluded=dict(Counter(r["reason"] for r in rows if r.get("reason"))))
    atomic_json(output / "collection.json", result)
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--product-manifest")
    parser.add_argument("--collection-config", default="configs/data/collection/retrieval.py")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    report = collect(**vars(args))
    if report["status"] != "completed":
        raise SystemExit("Collection incomplete; inspect errors.json and resume")
