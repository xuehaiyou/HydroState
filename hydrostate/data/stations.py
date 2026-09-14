from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ALIASES = {
    "site_id": ("site_id", "site", "siteid", "station", "station_id"),
    "latitude": ("latitude", "lat", "site_lat", "site_latitude", "location_lat"),
    "longitude": ("longitude", "lon", "long", "site_lon", "site_longitude",
                  "location_lon", "location_long"),
    "start": ("start", "start_date", "first_timestamp", "data_start"),
    "end": ("end", "end_date", "last_timestamp", "data_end"),
}


def find_column(frame: pd.DataFrame, name: str, required: bool = True) -> str | None:
    lookup = {str(c).strip().lower(): str(c) for c in frame.columns}
    for alias in ALIASES[name]:
        if alias in lookup:
            return lookup[alias]
    if required:
        raise ValueError(f"Cannot find {name}; available columns: {list(frame.columns)}")
    return None


def load_area(path: Path):
    from shapely.geometry import shape
    from shapely.ops import unary_union

    document = json.loads(path.read_text(encoding="utf-8"))
    return unary_union([shape(feature["geometry"]) for feature in document["features"]])


def filter_station_table(frame: pd.DataFrame, area_path: Path,
                         start: str | None = None, end: str | None = None) -> pd.DataFrame:
    from shapely.geometry import Point

    lat, lon = find_column(frame, "latitude"), find_column(frame, "longitude")
    area = load_area(area_path)
    valid = [area.covers(Point(float(x), float(y))) if pd.notna(x) and pd.notna(y) else False
             for x, y in zip(frame[lon], frame[lat], strict=True)]
    result = frame.loc[valid].copy()
    start_col, end_col = find_column(frame, "start", False), find_column(frame, "end", False)
    if start and end_col:
        result = result[pd.to_datetime(result[end_col], errors="coerce") >= pd.Timestamp(start)]
    if end and start_col:
        result = result[pd.to_datetime(result[start_col], errors="coerce") <= pd.Timestamp(end)]
    return result


def scalar(value):
    """Unwrap metadata values used by different ismn package releases."""
    for attribute in ("val", "value"):
        if hasattr(value, attribute):
            return getattr(value, attribute)
    return value
