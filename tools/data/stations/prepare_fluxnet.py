#!/usr/bin/env python
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import pandas as pd

from hydrostate.data.stations import filter_station_table, find_column


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover/filter/download FLUXNET Shuttle sites.")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--eligible-area", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--start", help="Optional start date; default: all available years")
    parser.add_argument("--end", help="Optional end date; default: all available years")
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    command = shutil.which("fluxnet-shuttle")
    if not command:
        raise RuntimeError("Install FLUXNET Shuttle; see docs/DATA_PREPARATION.md")

    snapshot = args.snapshot
    if snapshot is None:
        before = set(args.work_dir.glob("*.csv"))
        subprocess.run([command, "--quiet", "listall"], cwd=args.work_dir, check=True)
        candidates = list(set(args.work_dir.glob("*.csv")) - before) or list(
            args.work_dir.glob("*.csv"))
        if not candidates:
            raise RuntimeError("FLUXNET Shuttle did not create a snapshot CSV")
        snapshot = max(candidates, key=lambda p: p.stat().st_mtime)

    frame = pd.read_csv(snapshot)
    selected = filter_station_table(frame, args.eligible_area, args.start, args.end)
    output = args.work_dir / "selected_fluxnet_sites.csv"
    selected.to_csv(output, index=False)
    site_col = find_column(selected, "site_id")
    sites = sorted(selected[site_col].dropna().astype(str).unique())
    (args.work_dir / "selected_fluxnet_site_ids.txt").write_text("\n".join(sites) + "\n")
    print(f"selected {len(sites)} sites -> {output}")
    if args.download and sites:
        destination = args.work_dir / "downloads"
        destination.mkdir(exist_ok=True)
        subprocess.run([command, "download", "-f", str(snapshot), "-s", *sites,
                        "-o", str(destination)], check=True)


if __name__ == "__main__":
    main()
