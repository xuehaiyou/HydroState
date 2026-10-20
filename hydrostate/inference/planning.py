"""Fixed regional grids, PML-aligned dates and descriptive training overlap."""

from __future__ import annotations

import json
import math
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from affine import Affine
from pyproj import Transformer
from rasterio.features import geometry_mask
from shapely.geometry import box, mapping, shape
from shapely.ops import transform

from hydrostate.data.gee.contracts import (
    atomic_json,
    fingerprint,
    map_point,
)
from hydrostate.data.gee.contracts import input_shape as input_shape
from hydrostate.data.gee.contracts import input_transform as input_transform
from hydrostate.storage.packs import file_hash


def periods(start, end):
    """Annual 8-day supports; select by midpoint, end date exclusive.

    The first selected support can start before the requested first date.
    Its full bounds are retained, rather than silently changing ET semantics.
    """
    start, end = date.fromisoformat(start), date.fromisoformat(end)
    if end <= start:
        raise ValueError("end must follow start")
    result = []
    for year in range(start.year, end.year + 1):
        a, stop = date(year, 1, 1), date(year + 1, 1, 1)
        while a < stop:
            b = min(a + timedelta(days=8), stop)
            t = a + (b - a) // 2
            if start <= t < end:
                result.append(dict(time=str(t), start=str(a), end=str(b)))
            a = b
    return result


def make_grid(geometry, crs, core=32, stride=None):
    if core < 1 or core > 40 or (40 - core) % 2:
        raise ValueError("core must leave an even halo within the 40-pixel output")
    stride = core if stride is None else stride
    if stride < 1 or stride > core or core % stride:
        raise ValueError("stride must divide the retained core")
    a, b, c, d = geometry.bounds
    left, top = math.floor(a / 30) * 30, math.ceil(d / 30) * 30
    width = max(core, math.ceil((c - left) / (30 * stride)) * stride)
    height = max(core, math.ceil((top - b) / (30 * stride)) * stride)
    return dict(
        crs=crs,
        left=left,
        top=top,
        width=width,
        height=height,
        core=core,
        stride=stride,
        halo=(40 - core) // 2,
    )


def output_transform(grid):
    return Affine(30, 0, grid["left"], 0, -30, grid["top"])


def aoi_mask(plan):
    grid = plan["grid"]
    return geometry_mask(
        [plan["geometry"]], (grid["height"], grid["width"]), output_transform(grid), invert=True
    )


def windows(plan):
    grid, geometry = plan["grid"], shape(plan["geometry"])
    tr, core = output_transform(grid), grid["core"]
    rows = []
    stride = grid.get("stride", core)
    for row in range(0, grid["height"] - core + 1, stride):
        for col in range(0, grid["width"] - core + 1, stride):
            left, top = map_point(tr, col, row)
            if not geometry.intersects(box(left, top - core * 30, left + core * 30, top)):
                continue
            rows.append(dict(window_id=f"r{row:05d}_c{col:05d}", row=row, col=col))
    return rows


def window_geometry(grid, window):
    left, top = map_point(input_transform(grid), window["col"] * 3, window["row"] * 3)
    return box(left, top - 1200, left + 1200, top)


def training_records(manifest, crs, geometry):
    """Store only intersecting training footprints, with their actual dates."""
    table = pd.read_parquet(manifest)
    # Project only nearby records: opposite-hemisphere UTM coordinates can be
    # infinite and must never become apparent overlapping training footprints.
    reverse = Transformer.from_crs(crs, 4326, always_xy=True).transform
    west, south, east, north = transform(reverse, geometry).buffer(1).bounds
    table = table.loc[table.lon.between(west, east) & table.lat.between(south, north)]
    records = []
    for row in table.loc[table.split == "train"].to_dict("records"):
        if "transform" in row and "crs" in row:
            tr = row["transform"]
            tr = json.loads(tr) if isinstance(tr, str) else tr
            tr = Affine(*tr[:6])
            footprint = box(*map_point(tr, 0, 120), *map_point(tr, 120, 0))
            source = row["crs"]
        else:
            from hydrostate.data.gee_sampling import grid as sample_grid

            source, values = sample_grid(row["lon"], row["lat"])
            tr = Affine(*values)
            footprint = box(tr.c, tr.f - 1200, tr.c + 1200, tr.f)
        project = Transformer.from_crs(source, crs, always_xy=True).transform
        footprint = transform(project, footprint)
        if not footprint.is_valid or not all(math.isfinite(v) for v in footprint.bounds):
            raise ValueError(f"Invalid nearby training footprint: {row['sample_id']}")
        if footprint.intersects(geometry.buffer(1200)):
            records.append(
                dict(
                    sample_id=row["sample_id"],
                    time=str(row["target_time"])[:10],
                    geometry=mapping(footprint),
                )
            )
    return records


def overlap_records(grid, window, period, training, *, input_offsets=(-63, 1)):
    footprint = window_geometry(grid, window)
    t = date.fromisoformat(period["time"])
    a, b = date.fromisoformat(period["start"]), date.fromisoformat(period["end"])
    input_start, input_end = (t + timedelta(days=d) for d in input_offsets)
    spatial, context, target = [], [], []
    for record in training:
        if not footprint.intersects(shape(record["geometry"])):
            continue
        sid, day = record["sample_id"], date.fromisoformat(record["time"])
        spatial.append(sid)
        # The referenced training samples retain their original [day-63, day+1)
        # support. Compare full intervals when inference can also look forward.
        if day - timedelta(days=63) < input_end and input_start < day + timedelta(days=1):
            context.append(sid)
        if a <= day < b:
            target.append(sid)
    return dict(
        training_spatial_ids=spatial, training_context_ids=context, training_target_ids=target
    )


def initialize(root, config, manifest, *, pilot=False, ee=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path(config).read_text())
    identity = dict(config=config, pilot=pilot, training_manifest_sha256=file_hash(manifest))
    existing = root / "plan.json"
    if existing.exists():
        plan = json.loads(existing.read_text())
        if plan["identity"] != identity:
            raise ValueError("Existing plan differs; choose a new output directory")
        return plan
    if config["input_size"] != 120 or config["output_resolution_m"] != 30:
        raise ValueError("This checkpoint requires 120 pixels at 10 m, with 30 m outputs")
    if ee is None:
        import ee

        ee.Initialize(project=config["gee_project"])
    lake = (
        ee.FeatureCollection(config["lake_collection"])
        .filter(ee.Filter.eq("Hylak_id", config["lake_id"]))
        .first()
        .getInfo()
    )
    if lake["properties"]["Lake_name"] != config["lake_name"]:
        raise ValueError("Lake identifier/name mismatch")
    forward = Transformer.from_crs(4326, config["crs"], always_xy=True).transform
    geometry = transform(forward, shape(lake["geometry"])).buffer(config["buffer_m"])
    if pilot:
        x, y = forward(*config["pilot_center"])
        half = config["pilot_size_m"] / 2
        geometry = geometry.intersection(box(x - half, y - half, x + half, y + half))
    if geometry.is_empty:
        raise ValueError("Empty area")
    selected = periods(config["start"], config["end"])
    if pilot:
        selected = [
            min(
                selected,
                key=lambda p: abs((date.fromisoformat(p["time"]) - date.fromisoformat(t)).days),
            )
            for t in config["pilot_dates"]
        ]
        selected = sorted({p["time"]: p for p in selected}.values(), key=lambda p: p["time"])
    grid = make_grid(geometry, config["crs"], config["core_size"], config.get("stride_size"))
    training = training_records(manifest, config["crs"], geometry)
    plan = dict(
        version=1,
        identity=identity,
        config=config,
        grid=grid,
        periods=selected,
        geometry=mapping(geometry),
        area_km2=geometry.area / 1e6,
        training=training,
        boundary_note="HydroLAKES lake polygon plus fixed buffer; not a wetland inventory",
    )
    plan["plan_hash"] = fingerprint(plan)
    atomic_json(existing, plan)
    reverse = Transformer.from_crs(config["crs"], 4326, always_xy=True).transform
    atomic_json(
        root / "aoi.geojson",
        dict(
            type="Feature",
            geometry=mapping(transform(reverse, geometry)),
            properties=dict(
                name=config["name"],
                pilot=pilot,
                source=config["lake_collection"],
                lake_id=config["lake_id"],
                buffer_m=config["buffer_m"],
                area_km2=plan["area_km2"],
            ),
        ),
    )
    pd.DataFrame(windows(plan)).to_parquet(root / "windows.parquet", index=False)
    return plan
