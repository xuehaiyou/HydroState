"""Full temporal ET quality screening, independent of footprint metadata availability."""

import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from time import monotonic

import pandas as pd

from hydrostate.data.gee.contracts import atomic_json, fingerprint, locked

from .archives import sha256
from .availability import availability_rows
from .readers import UnsupportedSource, standard_flux_windows
from .temporal import epoch


def indexed_flux_files(index, splits, strata, sources):
    files = pd.read_parquet(Path(index) / "source_files.parquet")
    files = files[files.source.isin(sources)].copy()
    locations = pd.read_parquet(Path(splits) / "station_splits.parquet").merge(
        pd.read_parquet(Path(strata) / "station_strata.parquet")[
            ["location_id", "frequency_bin", "landcover", "climate_class"]
        ],
        on="location_id",
        validate="one_to_one",
    )
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
                "landcover",
                "climate_class",
            ]
        ],
        on=["site_id", "lon", "lat"],
        how="left",
        validate="many_to_one",
    )
    return files


def _read_et_file(task):
    row, root, config = task
    began = monotonic()
    path = Path(row["source_path"])
    signature = dict(bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns)
    if signature != dict(bytes=row["source_bytes"], mtime_ns=row["source_mtime_ns"]):
        raise ValueError(f"Source changed after indexing: {path}")
    if sha256(row["metadata_file"]) != row["metadata_sha256"]:
        raise ValueError(f"Metadata changed after indexing: {row['metadata_file']}")
    identity = fingerprint(dict(config=config, row=row, source_stat=signature))
    checkpoint = Path(root) / "labels" / (row["file_id"] + ".json")
    reused = checkpoint.exists()
    if reused:
        record = json.loads(checkpoint.read_text())
        if record["identity"] != identity:
            raise ValueError("ET checkpoint changed; use a new output directory")
    else:
        offset = pd.Timedelta(hours=row["utc_offset_hours"])
        # Include partially observed edge windows: the ordinary coverage gates decide.
        first = pd.Timestamp(row["time_start"]) - offset - pd.Timedelta(days=7)
        last = pd.Timestamp(row["time_end"]) - offset + pd.Timedelta(days=8)
        start = max(config["input_start"], first.date().isoformat())
        end = min(config["as_of"][:10], last.date().isoformat())
        meta = {
            k: row[k]
            for k in (
                "source",
                "site_id",
                "lon",
                "lat",
                "utc_offset_hours",
                "metadata_file",
                "metadata_sha256",
            )
        }
        meta.update(
            group_id=row["physical_group_id"],
            time_basis="local_standard",
            landcover=row["source_landcover"],
            support_type="footprint_unresolved",
        )
        try:
            labels = standard_flux_windows(path, meta, start, end, config["as_of"])
            valid = [r for r in labels if r["et"]["valid"]]
            rejected = Counter(reason for r in labels for reason in r["et"]["reasons"])
            record = dict(
                status="complete",
                windows=valid,
                calendar_windows=len(labels),
                temporal_valid=len(valid),
                rejection_counts=dict(rejected),
            )
        except UnsupportedSource as error:
            record = dict(
                status="unsupported",
                reason=str(error),
                windows=[],
                calendar_windows=0,
                temporal_valid=0,
                rejection_counts={},
            )
        record.update(
            identity=identity,
            source_path=str(path),
            source_stat=signature,
            elapsed_seconds=monotonic() - began,
        )
        atomic_json(checkpoint, record)
    candidates = []
    for i, label in enumerate(record["windows"]):
        candidates.append(
            dict(
                candidate_id=fingerprint([row["file_id"], label["target_time"], config["version"]])[
                    :24
                ],
                file_id=row["file_id"],
                label_record=str(checkpoint),
                record_index=i,
                **{
                    k: row[k]
                    for k in (
                        "source",
                        "site_id",
                        "physical_group_id",
                        "spatial_group_id",
                        "split",
                        "lon",
                        "lat",
                        "frequency_bin",
                        "landcover",
                        "climate_class",
                    )
                },
                target_time=label["target_time"],
                target_end=label["target_end"],
                year=int(label["target_time"][:4]),
                month=int(label["target_time"][5:7]),
                et=label["et"]["value"],
                valid_fraction=label["et"]["valid_fraction"],
                measured_fraction=label["et"]["measured_fraction"],
                day_fraction=label["et"]["day_fraction"],
                night_fraction=label["et"]["night_fraction"],
                et_corrected_valid=bool((label.get("et_corrected") or {}).get("valid")),
                spatial_support_verified=False,
                satellite_inputs_checked=False,
            )
        )
    status = {
        k: record.get(k)
        for k in (
            "status",
            "reason",
            "calendar_windows",
            "temporal_valid",
            "rejection_counts",
            "elapsed_seconds",
        )
    }
    status.update(
        file_id=row["file_id"],
        site_id=row["site_id"],
        source=row["source"],
        source_bytes=row["source_bytes"],
        reused_checkpoint=reused,
        current_elapsed_seconds=monotonic() - began,
    )
    return status, candidates


def prepare_et_candidates(
    index, splits, strata, output, *, as_of, sources=("fluxnet", "icos"), workers=2, limit=None
):
    index, splits, strata, output = map(Path, (index, splits, strata, output))
    if workers < 1 or limit is not None and limit < 1:
        raise ValueError("Positive worker/file limits required")
    sources = sorted(set(sources))
    if not sources or not set(sources) <= {"fluxnet", "icos"}:
        raise ValueError("Only standardized FLUXNET and ICOS are supported")
    files = indexed_flux_files(index, splits, strata, sources)
    input_start = json.loads((strata / "strata_report.json").read_text())["input_start"]
    exclusions = []
    rows = []
    for r in files.sort_values("file_id").to_dict("records"):
        reason = None
        columns = set(json.loads(r["columns"]))
        if r.get("metadata_issue") or pd.isna(r["physical_group_id"]):
            reason = "unresolved_spatial_or_time_metadata"
        elif not {"LE_F_MDS", "LE_F_MDS_QC"}.issubset(columns):
            reason = "missing_standard_ET_fields"
        else:
            offset = pd.Timedelta(hours=r["utc_offset_hours"])
            if pd.Timestamp(r["time_end"]) - offset <= pd.Timestamp(input_start) or (
                pd.Timestamp(r["time_start"]) - offset >= pd.Timestamp(epoch(as_of), unit="s")
            ):
                reason = "no_input_period_overlap"
        if reason:
            exclusions.append(dict(file_id=r["file_id"], site_id=r["site_id"], reason=reason))
            continue
        keep = [
            "file_id",
            "source",
            "site_id",
            "source_path",
            "source_bytes",
            "source_mtime_ns",
            "metadata_file",
            "metadata_sha256",
            "utc_offset_hours",
            "source_landcover",
            "time_start",
            "time_end",
            "physical_group_id",
            "spatial_group_id",
            "split",
            "lon",
            "lat",
        ]
        row = {k: r[k] for k in keep}
        row.update(
            {
                k: int(r[k]) if pd.notna(r[k]) else default
                for k, default in (("frequency_bin", -1), ("landcover", 0), ("climate_class", 0))
            }
        )
        rows.append(row)
    eligible_files = len(rows)
    if limit:
        # File-size quantiles make a small timing probe cover long and short records.
        ordered = sorted(rows, key=lambda r: (r["source_bytes"], r["file_id"]))
        count = min(limit, len(ordered))
        positions = sorted(
            {
                round(i * (len(ordered) - 1) / max(1, count - 1))
                for i in range(count)
            }
        )
        rows = [ordered[i] for i in positions]
    config = dict(
        version="site_standard_et_candidates_v1",
        as_of=as_of,
        input_start=input_start,
        sources=sources,
        qc="ONEFlux_HH_HR_0_1",
        edge_windows="overlap_then_QC",
        inputs_sha256={
            str(p): sha256(p)
            for p in (
                index / "source_files.parquet",
                splits / "station_splits.parquet",
                strata / "station_strata.parquet",
            )
        },
    )
    (output / "labels").mkdir(parents=True, exist_ok=True)
    with locked(output / ".candidates.lock", blocking=False):
        config_path = output / "config.json"
        if config_path.exists() and json.loads(config_path.read_text()) != config:
            raise ValueError("ET production inputs changed; use a new output directory")
        atomic_json(config_path, config)
        began = monotonic()
        all_rows, statuses = [], []
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
            tasks = [(r, str(output), config) for r in rows]
            for n, (status, candidates) in enumerate(pool.map(_read_et_file, tasks), 1):
                all_rows.extend(candidates)
                statuses.append(status)
                progress = dict(
                    done=n,
                    total=len(rows),
                    valid_windows=len(all_rows),
                    elapsed_seconds=monotonic() - began,
                )
                atomic_json(output / "progress.json", progress)
                if n % 10 == 0 or n == len(rows) or limit:
                    print(json.dumps(dict(progress, last=status)), flush=True)
        table = pd.DataFrame(all_rows)
        table.to_parquet(output / "candidates.parquet", index=False)
        frequency = availability_rows(table, "ET_temporal")
        pd.DataFrame(frequency).to_csv(output / "frequency_availability.csv", index=False)
        source_summary = []
        for source in sources:
            part = table[table.source.eq(source)] if not table.empty else table
            source_summary.append(
                dict(
                    source=source,
                    valid_source_windows=len(part),
                    physical_stations=int(part.physical_group_id.nunique())
                    if not part.empty
                    else 0,
                    station_windows=len(part.drop_duplicates(["physical_group_id", "target_time"]))
                    if not part.empty
                    else 0,
                )
            )
        report = dict(
            complete=len(rows) == eligible_files,
            config=config,
            indexed_files=len(files),
            eligible_files=eligible_files,
            processed_files=len(rows),
            exclusions=exclusions,
            processing_limit=limit,
            files=statuses,
            source_summary=source_summary,
            frequency=frequency,
            total_source_bytes=sum(r["source_bytes"] for r in rows),
            newly_read_source_bytes=sum(
                r["source_bytes"] for r in statuses if not r["reused_checkpoint"]
            ),
            candidates_sha256=sha256(output / "candidates.parquet"),
            elapsed_seconds=monotonic() - began,
            valid_source_windows=len(table),
            unknown_frequency_windows=int(table.frequency_bin.lt(0).sum())
            if not table.empty
            else 0,
            scope="Temporal label candidates; spatial support and satellite pairing pending",
        )
        atomic_json(output / "report.json", report)
        return report
