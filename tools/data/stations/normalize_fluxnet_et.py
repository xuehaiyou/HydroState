#!/usr/bin/env python
from __future__ import annotations

import argparse
import re
import zipfile
from io import TextIOWrapper
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def choose(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    return next((name for name in candidates if name in columns), None)


def frames(root: Path):
    for archive in root.rglob("*.zip"):
        with zipfile.ZipFile(archive) as zipped:
            names = [n for n in zipped.namelist() if n.endswith(".csv") and
                     ("_HH_" in n or "_HR_" in n)]
            for name in names:
                with zipped.open(name) as stream:
                    yield name, pd.read_csv(TextIOWrapper(stream))
    for path in root.rglob("*.csv"):
        if "_HH_" in path.name or "_HR_" in path.name:
            yield path.name, pd.read_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert FLUXNET latent heat to interval ET.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", help="Optional start date (inclusive)")
    parser.add_argument("--end", help="Optional end date (inclusive, entire day)")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer, count = None, 0
    try:
        for name, frame in frames(args.input):
            le = choose(list(frame), ("LE_F_MDS", "LE_CORR", "LE"))
            time = choose(list(frame), ("TIMESTAMP_START", "TIMESTAMP"))
            if not le or not time:
                continue
            site_match = re.search(r"(?:FLX_)?([A-Z]{2}-[A-Za-z0-9]+)", name)
            site = site_match.group(1) if site_match else name
            timestamp = pd.to_datetime(frame[time].astype(str), format="%Y%m%d%H%M",
                                       errors="coerce", utc=True)
            le_value = pd.to_numeric(frame[le], errors="coerce").replace(-9999, pd.NA)
            seconds = timestamp.diff().dt.total_seconds().median()
            seconds = float(seconds) if pd.notna(seconds) and seconds > 0 else 1800.0
            out = pd.DataFrame({"site_id": site, "time": timestamp,
                                "latent_heat_w_m2": le_value,
                                "et_mm_interval": le_value * seconds / 2.45e6,
                                "interval_seconds": seconds, "source_field": le})
            qc = choose(list(frame), ("LE_F_MDS_QC", "LE_RANDUNC"))
            out["quality"] = (pd.to_numeric(frame[qc], errors="coerce").astype(float)
                              if qc else float("nan"))
            if args.start:
                out = out[out.time >= pd.Timestamp(args.start, tz="UTC")]
            if args.end:
                stop = pd.Timestamp(args.end, tz="UTC") + pd.Timedelta(days=1)
                out = out[out.time < stop]
            out = out.dropna(subset=["time", "et_mm_interval"])
            if out.empty:
                continue
            table = pa.Table.from_pandas(out, preserve_index=False)
            writer = writer or pq.ParquetWriter(args.output, table.schema)
            writer.write_table(table)
            count += len(out)
    finally:
        if writer:
            writer.close()
    print(f"normalized {count} ET observations -> {args.output}")


if __name__ == "__main__":
    main()
