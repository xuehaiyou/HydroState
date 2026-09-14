#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from hydrostate.data.stations import load_area, scalar


def get(meta: dict, *names: str):
    lowered = {}
    for key, value in meta.items():
        if isinstance(key, tuple):
            if len(key) > 1 and str(key[1]).lower() != "val":
                continue
            key = key[0]
        lowered[str(key).lower()] = scalar(value)
    for name in names:
        if name in lowered:
            return lowered[name]
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize a user-downloaded ISMN archive.")
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--eligible-area", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", help="Optional start date (inclusive)")
    parser.add_argument("--end", help="Optional end date (inclusive)")
    parser.add_argument("--min-depth", type=float, default=0.0)
    parser.add_argument("--max-depth", type=float, default=0.1)
    args = parser.parse_args()

    from ismn.interface import ISMN_Interface
    from shapely.geometry import Point

    args.output.mkdir(parents=True, exist_ok=True)
    area = load_area(args.eligible_area)
    interface = ISMN_Interface(str(args.archive), parallel=False)
    ids = interface.get_dataset_ids("soil_moisture", min_depth=args.min_depth,
                                    max_depth=args.max_depth)
    station_rows, writer = [], None
    try:
        for dataset_id in ids:
            meta = interface.read_metadata(dataset_id, format="dict")
            lat = get(meta, "latitude", "lat")
            lon = get(meta, "longitude", "lon")
            if lat is None or lon is None or not area.covers(Point(float(lon), float(lat))):
                continue
            ts = interface.read_ts(dataset_id).copy()
            ts.index = pd.to_datetime(ts.index, utc=True)
            ts = ts.loc[args.start:args.end]
            if ts.empty:
                continue
            value_col = "soil_moisture" if "soil_moisture" in ts else ts.columns[0]
            obs = pd.DataFrame(
                {"dataset_id": str(dataset_id), "time": ts.index,
                 "soil_moisture_m3_m3": pd.to_numeric(ts[value_col], errors="coerce")}
            )
            obs["quality_flag"] = pd.Series(pd.NA, index=obs.index, dtype="string")
            for col in ts.columns:
                if "flag" in str(col).lower():
                    obs["quality_flag"] = ts[col].astype("string").to_numpy()
                    break
            obs = obs.dropna(subset=["soil_moisture_m3_m3"])
            if obs.empty:
                continue
            table = pa.Table.from_pandas(obs, preserve_index=False)
            writer = writer or pq.ParquetWriter(args.output / "observations.parquet", table.schema)
            writer.write_table(table)
            station_rows.append({"dataset_id": str(dataset_id), "latitude": float(lat),
                                 "longitude": float(lon), "network": get(meta, "network"),
                                 "station": get(meta, "station"),
                                 "depth_from_m": get(meta, "depth_from"),
                                 "depth_to_m": get(meta, "depth_to"), "n_obs": len(obs)})
    finally:
        if writer:
            writer.close()
    pd.DataFrame(station_rows).to_parquet(args.output / "stations.parquet", index=False)
    print(f"selected {len(station_rows)} ISMN series -> {args.output}")


if __name__ == "__main__":
    main()
