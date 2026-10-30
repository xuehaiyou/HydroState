"""Current regional input contract and shared native-resolution block grid."""

import json
import math
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd
from pyproj import CRS, Transformer
from shapely.geometry import box, mapping, shape
from shapely.ops import transform

from hydrostate.data.gee.contracts import atomic_json, fingerprint
from hydrostate.data.retrieval.contract import (
    INPUT_GROUPS,
    PROFILE,
    VERSION,
    intervals,
    product_end,
)

REGION_VERSION = "regional_composite_v1"


def epoch(day):
    return int(datetime.fromisoformat(day).replace(tzinfo=UTC).timestamp())


def load_plan(root):
    plan = json.loads((Path(root) / "plan.json").read_text())
    if (
        plan["version"] != REGION_VERSION
        or plan["input_profile"] != PROFILE
        or plan["input_contract"] != VERSION
    ):
        raise ValueError("Use a new regional root for the current five-slot input contract")
    expected = plan["identity"]
    if fingerprint({k: v for k, v in plan.items() if k != "identity"}) != expected:
        raise ValueError("Regional plan was modified after initialization")
    return plan


def window_box(plan, row, col):
    x, y = plan["grid"]["left"] + col * 10, plan["grid"]["top"] - row * 10
    return box(x, y - 1200, x + 1200, y)


def blocks_for_window(plan, row, col):
    size = plan["block_size"]
    return [
        (y, x)
        for y in range(row // size, (row + 119) // size + 1)
        for x in range(col // size, (col + 119) // size + 1)
    ]


def block_grid(plan, block, resolution=10):
    y, x = block
    size, grid = plan["block_size"], plan["grid"]
    height = min(size, grid["height"] - y * size) * 10 // resolution
    width = min(size, grid["width"] - x * size) * 10 // resolution
    return dict(
        crs=grid["crs"],
        height=height,
        width=width,
        transform=[
            resolution,
            0,
            grid["left"] + x * size * 10,
            0,
            -resolution,
            grid["top"] - y * size * 10,
        ],
    )


def product(plan, group, slot, block, orbit=None):
    value = dict(
        plan=plan["identity"],
        group=group,
        slot=list(slot),
        block=list(block),
        orbit=list(orbit) if orbit is not None else None,
    )
    return dict(value, id=fingerprint(value)[:32])


def initialize(root, config, *, ee=None):
    """Bind pixel semantics independently of the selected transport/runtime limits."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    config = dict(config)
    runtime = config.pop("drive")
    pixels = config.pop("pixels", None)
    if (root / "plan.json").exists():
        existing = load_plan(root)
        if existing["config"] != config:
            raise ValueError("Existing regional configuration differs; choose a new root")
        atomic_json(root / "drive.json", runtime)
        if pixels is not None:
            atomic_json(root / "pixels.json", pixels)
        return existing
    required = {
        "name",
        "project",
        "crs",
        "aoi",
        "start",
        "end",
        "block_size",
        "core_size",
        "stride",
        "min_active_slots",
        "min_input_fraction",
    }
    if set(config) != required:
        raise ValueError(f"Region configuration requires exactly {sorted(required)}")
    crs = CRS.from_user_input(config["crs"])
    if not crs.is_projected or any(a.unit_name != "metre" for a in crs.axis_info):
        raise ValueError("Regional grid requires a projected CRS in metres")
    core, stride, size = config["core_size"], config["stride"], config["block_size"]
    if (
        any(type(v) is not int for v in (core, stride, size))
        or not 2 <= core <= 40
        or (40 - core) % 2
        or stride <= 0
        or stride > core
        or stride % 2
        or size < 120
        or size % 6
    ):
        raise ValueError("Grid must align 10/15/20/30/60 m bands and overlapping 40-pixel outputs")
    if not 1 <= config["min_active_slots"] <= 5 or not 0 < config["min_input_fraction"] <= 1:
        raise ValueError("Invalid input support thresholds")
    aoi = config["aoi"]
    if "bbox" in aoi:
        geometry = box(*aoi["bbox"])
    elif "geojson" in aoi:
        record = json.loads(Path(aoi["geojson"]).read_text())
        geometry = shape(record.get("geometry", record))
    else:
        if ee is None:
            from hydrostate.data.gee.client import initialize as connect

            ee = connect(config["project"])
        record = (
            ee.FeatureCollection(aoi["collection"])
            .filter(ee.Filter.eq(aoi["id_field"], aoi["id"]))
            .first()
            .getInfo()
        )
        if not record:
            raise ValueError("AOI feature not found")
        geometry = shape(record["geometry"])
    geometry = transform(Transformer.from_crs(4326, crs, always_xy=True).transform, geometry)
    geometry = geometry.buffer(aoi.get("buffer_m", 0))
    if geometry.is_empty or not geometry.is_valid:
        raise ValueError("Invalid or empty AOI")
    left, bottom, right, top = geometry.bounds
    # Shared 60 m origin makes every window read exactly the training band grids.
    halo = (40 - core) * 15
    left, top = math.floor((left - halo) / 60) * 60, math.ceil((top + halo) / 60) * 60
    step = stride * 3
    width = 120 + max(0, math.ceil(((right + halo - left) / 10 - 120) / step)) * step
    height = 120 + max(0, math.ceil(((top - bottom + halo) / 10 - 120) / step)) * step
    periods = []
    start, end = date.fromisoformat(config["start"]), date.fromisoformat(config["end"])
    if end <= start:
        raise ValueError("end must follow start; selected target start dates are [start,end)")
    for year in range(start.year, end.year + 1):
        t = date(year, 1, 1)
        while t.year == year:
            if start <= t < end:
                periods.append(dict(start=str(t), end=product_end(str(t)), slots=intervals(str(t))))
            t += timedelta(days=8)
    if not periods:
        raise ValueError("No PML-aligned target start dates in the requested interval")
    from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.data.normalize import load_computed_config

    norm = load_computed_config()
    plan = dict(
        version=REGION_VERSION,
        input_contract=VERSION,
        input_profile=PROFILE,
        config=config,
        normalization=norm,
        grid=dict(crs=crs.to_string(), left=left, top=top, height=height, width=width),
        geometry=mapping(geometry),
        block_size=size,
        periods=periods,
        groups={
            k: dict(sensor=v.sensor, resolution=v.resolution, bands=list(v.bands))
            for k, v in INPUT_GROUPS.items()
        },
    )
    plan["identity"] = fingerprint(plan)
    rows = []
    margin = (40 - core) // 2 * 3
    for row in range(0, height - 119, step):
        for col in range(0, width - 119, step):
            footprint = window_box(plan, row, col).buffer(-margin * 10)
            if geometry.intersects(footprint):
                rows.append(dict(row=row, col=col))
    pd.DataFrame(rows).to_parquet(root / "windows.parquet", index=False)
    geographic = transform(Transformer.from_crs(crs, 4326, always_xy=True).transform, geometry)
    atomic_json(
        root / "aoi.geojson",
        dict(type="Feature", geometry=mapping(geographic), properties=dict(name=config["name"])),
    )
    atomic_json(root / "drive.json", runtime)
    if pixels is not None:
        atomic_json(root / "pixels.json", pixels)
    atomic_json(root / "plan.json", plan)
    # Match the canonical JSON types used by every resumed run and frozen target.
    return load_plan(root)
