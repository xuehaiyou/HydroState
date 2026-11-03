"""Count physical station windows and 30-day capacity, without implying input acceptance."""

from pathlib import Path

import pandas as pd

from hydrostate.data.gee.contracts import atomic_json

from .archives import sha256
from .candidates import spaced_capacity


def availability_rows(table, target):
    rows = []
    for split in ("all", "train", "validation", "test"):
        subset = table if split == "all" or table.empty else table[table.split.eq(split)]
        for frequency in range(7):
            part = subset if subset.empty else subset[subset.frequency_bin.eq(frequency)]
            rows.append(
                dict(
                    target=target,
                    split=split,
                    frequency_bin=frequency,
                    physical_stations=0 if part.empty else part.physical_group_id.nunique(),
                    station_windows=0
                    if part.empty
                    else len(part.drop_duplicates(["physical_group_id", "target_time"])),
                    capacity_30d=0
                    if part.empty
                    else sum(
                        spaced_capacity(g.target_time) for _, g in part.groupby("physical_group_id")
                    ),
                )
            )
    return rows


def summarize_availability(sm_candidates, et_candidates, output):
    sm_candidates, et_candidates, output = map(Path, (sm_candidates, et_candidates, output))
    output.mkdir(parents=True, exist_ok=True)
    sm, et = pd.read_parquet(sm_candidates), pd.read_parquet(et_candidates)
    keys = ["physical_group_id", "target_time", "split", "frequency_bin"]
    combined = pd.concat([t[keys] for t in (sm, et) if not t.empty], ignore_index=True)
    if combined.groupby("physical_group_id").split.nunique().gt(1).any():
        raise ValueError("Physical station appears in multiple splits")
    # Never sum frequency capacities if a physical group spans bins: gap applies across bins.
    mixed_bins = int(combined.groupby("physical_group_id").frequency_bin.nunique().gt(1).sum())
    rows = availability_rows(sm, "SM_label") + availability_rows(et, "ET_label_and_footprint")
    rows += availability_rows(combined, "union_label_and_support")
    pd.DataFrame(rows).to_csv(output / "frequency_availability.csv", index=False)
    result = dict(
        complete=True,
        rows=rows,
        physical_groups_spanning_bins=mixed_bins,
        union_station_windows=len(combined.drop_duplicates(["physical_group_id", "target_time"])),
        union_capacity_30d=sum(
            spaced_capacity(g.target_time) for _, g in combined.groupby("physical_group_id")
        ),
        source_sha256={str(p): sha256(p) for p in (sm_candidates, et_candidates)},
        sampling_priority=["frequency_bin", "landcover_climate", "year", "month"],
        scope="Candidate capacity, not satellite-validated training samples; ET is ICOS only",
        note="30d counts are maxima, not a stratified random selection; unavailable bins stay zero",
    )
    atomic_json(output / "availability.json", result)
    return result
