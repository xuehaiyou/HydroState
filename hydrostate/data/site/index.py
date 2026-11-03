"""Header-only source index: observations, QC eligibility and spatial identity stay distinct."""

import csv
import json
import re
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from hydrostate.data.gee.contracts import atomic_json, fingerprint, locked

from .archives import sha256
from .readers import UnsupportedSource, ismn_header

INDEX_VERSION = "site_source_index_v2"


def bif_rows(path):
    """Reuse the Site_Process BIF CSV/XLSX convention without opening its database."""
    if path.suffix.lower() == ".xlsx":
        from openpyxl import load_workbook

        book = load_workbook(path, read_only=True, data_only=True)
        try:
            rows = book.active.iter_rows(values_only=True)
            header = next(rows)
            return [dict(zip(header, row, strict=True)) for row in rows]
        finally:
            book.close()
    with path.open(encoding="utf-8-sig", newline="", errors="strict") as stream:
        return list(csv.DictReader(stream))


def bif_metadata(path, source, station):
    rows = bif_rows(path)

    def unique(name):
        values = {str(r.get("DATAVALUE", "")).strip() for r in rows if r.get("VARIABLE") == name}
        return sorted(values - {"", "None", "-9999", "-9999.0"})

    result = dict(
        source=source,
        site_id=f"{source}:{station}",
        group_id=f"flux:{station}",
        network=source,
        metadata_file=str(path),
        metadata_sha256=sha256(path),
        support_type="footprint_unresolved",
        time_basis="local_standard",
    )
    problems = []
    for name, key, bounds in (
        ("LOCATION_LAT", "lat", (-90, 90)),
        ("LOCATION_LONG", "lon", (-180, 180)),
        ("UTC_OFFSET", "utc_offset_hours", (-14, 14)),
    ):
        values = unique(name)
        numeric = sorted({float(v) for v in values})
        result[key + "_history"] = json.dumps(values)
        if len(numeric) == 1 and bounds[0] <= numeric[0] <= bounds[1]:
            result[key] = numeric[0]
        else:
            result[key] = None
            problems.append(f"unresolved_{name}")
    result["source_landcover"] = ";".join(unique("IGBP"))
    result["metadata_issue"] = ";".join(problems)
    return result


def file_edges(path, *, ismn=False):
    """Read the first record and bounded tail, never scan a time series for inventory."""
    with path.open(encoding="utf-8-sig") as stream:
        if ismn:
            next(stream)
            header = []
        else:
            line = next(stream)
            while line.startswith("#"):
                line = next(stream)
            header = next(csv.reader([line]))
        first = next(stream, "").strip()
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - 131072))
        tail = stream.read().decode("utf-8-sig").strip().splitlines()
    if not first or not tail:
        raise UnsupportedSource("No observation records")
    if ismn:

        def parse(row):
            return datetime.strptime(" ".join(row.split()[:2]), "%Y/%m/%d %H:%M")

        start, end = parse(first), parse(tail[-1]) + timedelta(minutes=1)
    else:
        if header[:2] != ["TIMESTAMP_START", "TIMESTAMP_END"]:
            raise UnsupportedSource("No explicit HH/HR interval timestamps")
        start = datetime.strptime(first.split(",")[0], "%Y%m%d%H%M")
        end = datetime.strptime(tail[-1].split(",")[1], "%Y%m%d%H%M")
    if end <= start:
        raise ValueError(f"Reversed source date bounds: {path}")
    return header, start.isoformat(), end.isoformat()


def source_tables(root, source):
    if source == "ismn":
        return sorted(root.glob("*/*/*/*_sm_*.stm"))
    if source == "fluxnet":
        return sorted(
            p
            for p in root.glob("*/*.csv")
            if re.search(r"_FLUXNET_(?:FLUXMET_|FULLSET_)?(?:HH|HR)_", p.name)
        )
    if source == "ameriflux":
        return sorted(p for p in root.glob("*/*.csv") if re.search(r"_BASE_(?:HH|HR)_", p.name))
    # ARCHIVE already contains standalone FLUXES/METEO products: index it once.
    return sorted(
        p for p in root.glob("*/*_ARCHIVE_L2/*_FLUXNET_HH_L2.csv") if "_VARINFO_" not in p.name
    )


def source_record(path, source, metadata_cache):
    stat = path.stat()
    result = dict(
        source=source,
        source_path=str(path.resolve()),
        source_bytes=stat.st_size,
        source_mtime_ns=stat.st_mtime_ns,
        file_id=fingerprint(str(path.resolve()))[:24],
    )
    if source == "ismn":
        result.update(ismn_header(path, require_shallow=False))
        try:
            ismn_header(path)
            result.update(sm_header_eligible=True, eligibility_reason="")
        except UnsupportedSource as error:
            result.update(sm_header_eligible=False, eligibility_reason=str(error))
        result.update(et_header_eligible=False, metadata_issue="")
    else:
        station_match = re.search(r"_(\w{2}-[A-Za-z0-9]+)_", path.name)
        if station_match is None:
            raise UnsupportedSource("Cannot identify flux station code")
        station = station_match[1]
        result.update(site_id=f"{source}:{station}", group_id=f"flux:{station}")
        pattern = "*_SITEINFO_*.csv" if source == "icos" else "*_BIF_*"
        metadata = sorted(p for p in path.parent.glob(pattern) if p.suffix in (".csv", ".xlsx"))
        if len(metadata) != 1:
            raise UnsupportedSource("One BIF/SITEINFO metadata file required")
        key = str(metadata[0])
        if key not in metadata_cache:
            metadata_cache[key] = bif_metadata(metadata[0], source, station)
        result.update(metadata_cache[key])
        result.update(
            sm_header_eligible=False,
            et_header_eligible=False,
            eligibility_reason="flux_SM_depth_and_QC_mapping_pending;ET_support_pending",
        )
    columns, first, last = file_edges(path, ismn=source == "ismn")
    result.update(
        time_start=first,
        time_end=last,
        extent_basis="first_last_record_not_valid_QC",
        columns=json.dumps(columns),
        status="indexed",
    )
    if source != "ismn":
        result["et_header_eligible"] = (
            source == "fluxnet"
            and not result["metadata_issue"]
            and {"LE_F_MDS", "LE_F_MDS_QC"}.issubset(columns)
        )
        result["footprint_fields"] = json.dumps(
            [
                c
                for c in columns
                if re.fullmatch(r"(?:USTAR|WD|WS|MO_LENGTH|ZL|V_SIGMA|PBLH)(?:_\d+(?:_\d+)*)?", c)
            ]
        )
        result["has_sm_columns"] = any(c.startswith("SWC") for c in columns)
    return result


def build_index(raw_root, output, *, limit=None, reuse=None):
    raw_root, output = Path(raw_root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    config = dict(
        version=INDEX_VERSION,
        raw_root=str(raw_root.resolve()),
        limit=limit,
        reuse=str(Path(reuse).resolve()) if reuse else None,
    )
    with locked(output / ".index.lock", blocking=False):
        config_path = output / "index_config.json"
        if config_path.exists() and json.loads(config_path.read_text()) != config:
            raise ValueError("Source index scope changed; use a new directory")
        atomic_json(config_path, config)
        records, metadata_cache = [], {}
        cache = output / "files"
        cache.mkdir(exist_ok=True)
        for source in ("ismn", "fluxnet", "ameriflux", "icos"):
            paths = source_tables(raw_root / source, source)
            if not paths:
                raise ValueError(f"No {source} subdaily files found")
            if limit:
                paths = paths[:limit]
            for index, path in enumerate(paths, 1):
                dest = cache / f"{fingerprint(str(path.resolve()))[:24]}.json"
                previous = dest
                if not previous.exists() and reuse:
                    previous = Path(reuse) / "files" / dest.name
                if previous.exists():
                    row = json.loads(previous.read_text())
                    if (row["source_bytes"], row["source_mtime_ns"]) != (
                        path.stat().st_size,
                        path.stat().st_mtime_ns,
                    ):
                        raise ValueError(f"Indexed source changed: {path}")
                    if previous != dest:
                        atomic_json(dest, row)
                else:
                    try:
                        row = source_record(path, source, metadata_cache)
                    except UnsupportedSource as error:
                        row = dict(
                            source=source,
                            source_path=str(path),
                            status="unsupported",
                            eligibility_reason=str(error),
                            source_bytes=path.stat().st_size,
                            source_mtime_ns=path.stat().st_mtime_ns,
                        )
                    atomic_json(dest, row)
                records.append(row)
                if index % 500 == 0 or index == len(paths):
                    progress = dict(source=source, indexed=index, total=len(paths))
                    print(json.dumps(progress), flush=True)
        frame = pd.DataFrame(records)
        frame.to_parquet(output / "source_files.parquet", index=False)
        usable = frame[frame.status.eq("indexed")].copy()
        columns = ["site_id", "group_id", "source", "network", "lat", "lon", "metadata_issue"]
        stations = (
            usable[columns]
            .groupby(columns[:-1], dropna=False, sort=True)["metadata_issue"]
            .agg(lambda values: ";".join(sorted(set(values.dropna()) - {""})))
            .reset_index()
        )
        # More than one location is kept as separate geometry, but linked by source station ID.
        stations["location_id"] = [
            fingerprint([r.site_id, r.lon, r.lat])[:24] for r in stations.itertuples()
        ]
        stations.to_parquet(output / "stations.parquet", index=False)
        report = dict(
            complete=True,
            version=INDEX_VERSION,
            source_files=len(frame),
            source_counts=dict(Counter(frame.source)),
            locations=len(stations),
            source_site_ids=stations.site_id.nunique(),
            sm_header_eligible=int(frame.sm_header_eligible.fillna(False).sum()),
            et_header_eligible=int(frame.et_header_eligible.fillna(False).sum()),
            status_counts=dict(Counter(frame.status)),
            note="Header eligibility is not a count of quality-screened training windows",
        )
        atomic_json(output / "index_report.json", report)
        return report
