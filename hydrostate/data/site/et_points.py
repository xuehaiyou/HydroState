"""Small, reproducible ET diagnostics; temporal quality does not approve a tower footprint."""

import json
import random
import shutil
from collections import Counter
from pathlib import Path

import pandas as pd

from hydrostate.data.gee.contracts import atomic_json, fingerprint, locked

from .archives import sha256
from .prepare import LABEL_VERSION, prepare_labels
from .temporal import DAY, epoch


def select_et_windows(records, count, *, seed=43, gap_days=30):
    """Select by temporal quality and month diversity, never by product agreement."""
    rows = [r for r in records if r["et"]["valid"]]
    random.Random(seed).shuffle(rows)
    selected, months = [], Counter()
    while rows and len(selected) < count:
        row = min(rows, key=lambda r: months[r["target_time"][5:7]])
        selected.append(row)
        months[row["target_time"][5:7]] += 1
        rows = [
            r
            for r in rows
            if abs(epoch(r["target_time"]) - epoch(row["target_time"])) >= gap_days * DAY
        ]
    return selected


def prepare_et_points(
    index,
    splits,
    strata,
    output,
    *,
    start,
    end,
    as_of,
    stations=4,
    per_site=4,
    max_files=12,
    seed=43,
    reuse_labels=(),
):
    """Read a bounded number of FLUXNET files from fixed training groups on a CPU node."""
    if min(stations, per_site, max_files) < 1 or epoch(end) <= epoch(start):
        raise ValueError("Positive diagnostic counts and a nonempty date interval are required")
    index, splits, strata, output = map(Path, (index, splits, strata, output))
    output.mkdir(parents=True, exist_ok=True)
    config = dict(
        version="site_et_point_selection_v1",
        start=start,
        end=end,
        as_of=as_of,
        stations=stations,
        per_site=per_site,
        max_files=max_files,
        seed=seed,
        gap_days=30,
        index_sha256=sha256(index / "source_files.parquet"),
        splits_sha256=sha256(splits / "station_splits.parquet"),
        strata_sha256=sha256(strata / "station_strata.parquet"),
        reuse_labels=[str(Path(p)) for p in reuse_labels],
    )
    files = pd.read_parquet(index / "source_files.parquet")
    locations = pd.read_parquet(splits / "station_splits.parquet").merge(
        pd.read_parquet(strata / "station_strata.parquet")[
            [
                "location_id",
                "frequency_bin",
                "climate_class",
            ]
        ],
        on="location_id",
        validate="one_to_one",
    )
    files = files[files.source.eq("fluxnet") & files.et_header_eligible.fillna(False)]
    files = files.merge(
        locations[
            [
                "site_id",
                "lon",
                "lat",
                "physical_group_id",
                "spatial_group_id",
                "split",
                "frequency_bin",
                "climate_class",
            ]
        ],
        on=["site_id", "lon", "lat"],
        validate="many_to_one",
    )
    files = files[
        files.split.eq("train")
        & files.time_start.str[:10].lt(end)
        & files.time_end.str[:10].gt(start)
        & files.frequency_bin.ge(0)
        & files.source_landcover.notna()
        & ~files.source_landcover.isin(["unknown", "WAT", "SNO"])
    ]
    rows = files.fillna({"climate_class": 0}).sort_values("file_id").to_dict("records")
    random.Random(seed).shuffle(rows)
    label_config = dict(version=LABEL_VERSION, source="fluxnet", start=start, end=end, as_of=as_of)
    existing = {}
    for directory in map(Path, reuse_labels):
        for path in sorted(directory.glob("*.json")):
            value = json.loads(path.read_text())
            if value.get("config_sha256") == fingerprint(label_config):
                existing[value["source_path"]] = path
    with locked(output / ".selection.lock", blocking=False):
        path = output / "selection_config.json"
        if path.exists() and json.loads(path.read_text()) != config:
            raise ValueError("ET diagnostic definition changed; use a new output directory")
        atomic_json(path, config)
        selected, audit, groups, spatial_groups = [], [], set(), set()
        bins, classes = Counter(), Counter()
        while rows and len(groups) < stations and len(audit) < max_files:
            row = min(
                rows,
                key=lambda r: (
                    r["source_path"] not in existing,
                    bins[int(r["frequency_bin"])],
                    classes[(r["source_landcover"], r["climate_class"])],
                ),
            )
            rows.remove(row)
            if row["physical_group_id"] in groups or row["spatial_group_id"] in spatial_groups:
                continue
            source = Path(row["source_path"])
            if (source.stat().st_size, source.stat().st_mtime_ns) != (
                row["source_bytes"],
                row["source_mtime_ns"],
            ) or sha256(row["metadata_file"]) != row["metadata_sha256"]:
                raise ValueError(f"Source or BIF changed since indexing: {source}")
            labels = output / "labels" / row["file_id"]
            labels.mkdir(parents=True, exist_ok=True)
            checkpoint = labels / (fingerprint(str(source.resolve()))[:24] + ".json")
            reuse = existing.get(str(source))
            if reuse and not checkpoint.exists():
                shutil.copy2(reuse, checkpoint)
            prepare_labels([source], labels, "fluxnet", start, end, as_of)
            result = json.loads(checkpoint.read_text())
            chosen = select_et_windows(result["windows"], per_site, seed=seed + len(audit))
            for record in chosen:
                if (record["lon"], record["lat"]) != (row["lon"], row["lat"]):
                    raise ValueError("Parsed station position differs from fixed split")
                if record["metadata_sha256"] != row["metadata_sha256"]:
                    raise ValueError("Reused ET metadata differs from the indexed BIF")
                selected.append(
                    dict(
                        record,
                        split="train",
                        group_id=row["physical_group_id"],
                        spatial_group_id=row["spatial_group_id"],
                        frequency_bin=int(row["frequency_bin"]),
                        climate_class=int(row["climate_class"] or 0),
                        label_record=str(checkpoint),
                    )
                )
            audit.append(
                dict(
                    site_id=row["site_id"],
                    source_file=str(source),
                    status=result["status"],
                    reason=result.get("reason"),
                    temporal_windows=len(result["windows"]),
                    temporal_valid=sum(r["et"]["valid"] for r in result["windows"]),
                    selected=len(chosen),
                    reused_label=str(reuse) if reuse else None,
                )
            )
            if chosen:
                groups.add(row["physical_group_id"])
                spatial_groups.add(row["spatial_group_id"])
                bins[int(row["frequency_bin"])] += 1
                classes[(row["source_landcover"], row["climate_class"])] += 1
        report = dict(
            config=config,
            selected=selected,
            files=audit,
            stations=len(groups),
            windows=len(selected),
            requested_windows=stations * per_site,
            scope="temporal_quality_diagnostic_only; footprint_unresolved; no_training_release",
        )
        atomic_json(output / "point_selection.json", report)
        if not selected:
            raise ValueError("No temporally qualified ET diagnostic windows; see selection audit")
        return report
