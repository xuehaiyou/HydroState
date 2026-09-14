from __future__ import annotations

import json
import re
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd


@dataclass
class Scene:
    sensor: str
    scene_id: str
    tile_id: str
    acquired: str
    path: str
    crs: str
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float
    resolution_x: float
    resolution_y: float


def _date(text: str) -> str:
    match = re.search(r"(?:19|20)\d{6}", text)
    if not match:
        return ""
    return datetime.strptime(match.group(), "%Y%m%d").date().isoformat()


def _inspect(item: tuple[str, str]) -> dict | None:
    sensor, filename = item
    import rasterio
    from rasterio.warp import transform_bounds

    path = Path(filename)
    try:
        with rasterio.open(path) as src:
            if src.crs is None:
                return None
            bounds = transform_bounds(src.crs, "EPSG:4326", *src.bounds, densify_pts=21)
            rx, ry = src.res
            crs = src.crs.to_string()
    except Exception:
        return None

    if sensor == "s1":
        scene_dir = path.parent
        scene_id = scene_dir.name
        rel = scene_dir.parts
        tile = "".join(rel[-4:-1]) if len(rel) >= 4 else ""
    elif sensor == "s2":
        safe = next((p for p in path.parents if p.name.endswith(".SAFE")), path.parent)
        scene_dir, scene_id = safe, safe.name.removesuffix(".SAFE")
        match = re.search(r"T(\d{2}[A-Z]{3})", scene_id)
        tile = match.group(1) if match else ""
    else:
        scene_dir = path.parent
        scene_id = scene_dir.name
        parts = scene_id.split("_")
        tile = parts[2] if len(parts) > 2 else ""

    return asdict(
        Scene(sensor, scene_id, tile, _date(scene_id), str(scene_dir), crs,
              float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3]),
              float(rx), float(ry))
    )


def discover_rasters(sensor: str, root: Path) -> Iterable[Path]:
    if sensor == "s1":
        yield from root.rglob("Sigma0_VV_db.tif")
    elif sensor == "s2":
        # One 10 m band is sufficient to identify each SAFE footprint.
        yield from root.rglob("*_B02_10m.jp2")
    elif sensor == "landsat":
        yield from root.rglob("*_SR_B2.TIF")
    else:
        raise ValueError(f"Unsupported sensor: {sensor}")


def build_catalog(sensor: str, root: Path, output: Path, workers: int = 8,
                  limit: int | None = None) -> pd.DataFrame:
    paths = discover_rasters(sensor, root)
    if limit is not None:
        from itertools import islice
        paths = islice(paths, limit)
    output.parent.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = [row for row in pool.map(_inspect, ((sensor, str(p)) for p in paths),
                                        chunksize=16) if row]
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.drop_duplicates("scene_id").sort_values(["acquired", "scene_id"])
    frame.to_parquet(output, index=False)
    return frame


def build_common_area(catalogs: list[Path], output: Path,
                      start: str | None = None, end: str | None = None) -> None:
    from shapely.geometry import box, mapping
    from shapely.ops import unary_union

    areas = []
    counts = {}
    date_ranges = {}
    for path in catalogs:
        frame = pd.read_parquet(path)
        if frame.empty:
            raise RuntimeError(f"No scenes in catalog: {path}")
        if start:
            frame = frame[frame.acquired >= start]
        if end:
            frame = frame[frame.acquired <= end]
        if frame.empty:
            raise RuntimeError(f"No scenes in {start}..{end}: {path}")
        sensor = str(frame.sensor.iloc[0])
        counts[sensor] = len(frame)
        dates = frame.loc[frame.acquired.ne(""), "acquired"]
        date_ranges[sensor] = {"start": dates.min() if len(dates) else None,
                               "end": dates.max() if len(dates) else None}
        unique = frame.drop_duplicates(["min_lon", "min_lat", "max_lon", "max_lat"])
        areas.append(unary_union([box(r.min_lon, r.min_lat, r.max_lon, r.max_lat)
                                  for r in unique.itertuples()]))
    common = areas[0]
    for area in areas[1:]:
        common = common.intersection(area)
    if common.is_empty:
        raise RuntimeError("The three sensor coverages have no common area")
    feature = {"type": "Feature", "properties": {"start": start, "end": end,
               "scene_counts": counts, "sensor_date_ranges": date_ranges},
               "geometry": mapping(common)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"type": "FeatureCollection", "features": [feature]},
                                 ensure_ascii=False), encoding="utf-8")
