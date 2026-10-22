#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from hydrostate.data.coverage import build_catalog, build_common_area


def main() -> None:
    parser = argparse.ArgumentParser(description="Index local imagery and intersect footprints.")
    parser.add_argument("--s1", type=Path, required=True)
    parser.add_argument("--s2", type=Path, required=True)
    parser.add_argument("--landsat", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", help="Optional coverage start date (inclusive)")
    parser.add_argument("--end", help="Optional coverage end date (inclusive)")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    catalogs = []
    for sensor, root in (("s1", args.s1), ("s2", args.s2), ("landsat", args.landsat)):
        target = args.output / f"{sensor}_scenes.parquet"
        frame = build_catalog(sensor, root, target, args.workers, args.limit)
        print(f"{sensor}: {len(frame)} scenes -> {target}", flush=True)
        catalogs.append(target)
    target = args.output / "common_coverage.geojson"
    build_common_area(catalogs, target, args.start, args.end)
    print(f"common coverage -> {target}")


if __name__ == "__main__":
    main()
