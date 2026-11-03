"""Resumable ICOS ET/FFP candidates from already extracted L2 archives, without a DB."""

import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import timedelta
from pathlib import Path
from time import monotonic

import numpy as np
import pandas as pd
import zarr
from numcodecs import Blosc

from hydrostate.data.gee.contracts import atomic_json, fingerprint, locked

from .archives import sha256
from .footprint import parameter_valid, window_mass
from .readers import UnsupportedSource
from .temporal import DAY, epoch, interval_et, windows

PARAMETERS = dict(ws="WS", h="PBLH", ol="MO_LENGTH", sigmav="V_SIGMA", ustar="USTAR", wd="WD")


def history_values(events, times, *, max_age_days=None):
    """Last known metadata at/before observation; no future backfill or interpolation."""
    values = np.full(len(times), np.nan)
    if not events:
        return values
    ordered = sorted(events)
    dates, heights = np.asarray(ordered, float).T
    if len(set(dates)) != len(dates):
        raise ValueError("Conflicting metadata updates at the same timestamp")
    indices = np.searchsorted(dates, times, side="right") - 1
    valid = indices >= 0
    if max_age_days is not None:
        valid &= times - dates[np.maximum(indices, 0)] <= max_age_days * DAY
    values[valid] = heights[indices[valid]]
    return values


def metadata_events(path, kind, utc_offset):
    table = pd.read_csv(path, dtype=str).fillna("")
    group = "GRP_VAR_INFO" if kind == "instrument" else "GRP_HEIGHTC"
    table = table[table.VARIABLE_GROUP.eq(group)]
    events, discarded = {}, Counter()
    for _, rows in table.groupby("GROUP_ID", sort=False):
        if rows.VARIABLE.duplicated().any():
            raise ValueError(f"Duplicate metadata variable: {path}")
        r = dict(zip(rows.VARIABLE, rows.DATAVALUE, strict=True))
        if kind == "instrument":
            if r.get("VAR_INFO_VARNAME") != "USTAR":
                continue
            date, value, priority = r.get("VAR_INFO_DATE", ""), r.get("VAR_INFO_HEIGHT"), 0
        else:
            if r.get("HEIGHTC_SPP") or r.get("HEIGHTC_VEG_STATUS", "") not in ("", "Alive"):
                continue
            statistic = r.get("HEIGHTC_STATISTIC")
            if statistic not in ("Mean", "90th Percentile"):
                continue
            # A range observation is not used before its last measurement date.
            date = r.get("HEIGHTC_DATE") or r.get("HEIGHTC_DATE_END", "")
            value, priority = r.get("HEIGHTC"), int(statistic == "90th Percentile")
        if not value or value in ("-9999", "-9999.0") or len(date) not in (8, 12):
            discarded["missing_value_or_exact_date"] += 1
            continue
        stamp = pd.to_datetime(date, format="%Y%m%d" if len(date) == 8 else "%Y%m%d%H%M")
        time = int(stamp.timestamp()) - utc_offset * 3600
        number = float(value)
        if not np.isfinite(number) or number < 0:
            discarded["invalid_height"] += 1
            continue
        if time in events and events[time][0] == priority and events[time][1] != number:
            raise UnsupportedSource(f"Ambiguous {kind} height at {date}")
        if time not in events or priority > events[time][0]:
            events[time] = (priority, number)
    return [(t, v[1]) for t, v in sorted(events.items())], dict(discarded)


def read_intervals(path, columns, utc_offset):
    names = ["TIMESTAMP_START", "TIMESTAMP_END", *columns]
    header = pd.read_csv(path, nrows=0).columns
    missing = set(names) - set(header)
    if missing:
        raise UnsupportedSource(f"Missing interval fields: {sorted(missing)}")
    data = pd.read_csv(
        path,
        usecols=names,
        na_values=[-9999, -6999],
        dtype={"TIMESTAMP_START": str, "TIMESTAMP_END": str},
    )
    for key in names[:2]:
        data[key] = (
            pd.to_datetime(data[key], format="%Y%m%d%H%M").astype("datetime64[s]").astype("int64")
        )
        data[key] -= int(utc_offset * 3600)
    starts, ends = data.TIMESTAMP_START.to_numpy(), data.TIMESTAMP_END.to_numpy()
    if np.any(ends <= starts) or np.any(ends - starts > 3600) or np.any(starts[1:] < ends[:-1]):
        raise ValueError(f"Unordered/overlapping intervals: {path}")
    return data


def process_station(row, root, config):
    began = monotonic()
    station = row["site_id"].split(":", 1)[1]
    output = Path(root) / "stations" / station
    output.mkdir(parents=True, exist_ok=True)
    source = Path(row["source_path"])
    prefix = source.parent / f"ICOSETC_{station}"
    paths = {
        "fluxnet": source,
        **{
            k: Path(f"{prefix}_{v}_L2.csv")
            for k, v in (
                ("fluxes", "FLUXES"),
                ("instrument", "VARINFO_FLUXES"),
                ("canopy", "ANCILLARY"),
                ("siteinfo", "SITEINFO"),
            )
        },
    }
    if any(not p.exists() for p in paths.values()):
        raise UnsupportedSource(f"Incomplete archive companions: {station}")
    signatures = {
        k: dict(path=str(p), bytes=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns)
        for k, p in paths.items()
    }
    if (source.stat().st_size, source.stat().st_mtime_ns) != (
        row["source_bytes"],
        row["source_mtime_ns"],
    ) or sha256(paths["siteinfo"]) != row["metadata_sha256"]:
        raise ValueError(f"Source changed after fixed split/index: {source}")
    for key in ("instrument", "canopy", "siteinfo"):
        signatures[key]["sha256"] = sha256(paths[key])
    identity = fingerprint(dict(config=config, sources=signatures, row=row))
    checkpoint = output / "report.json"
    if checkpoint.exists():
        report = json.loads(checkpoint.read_text())
        if report["identity"] != identity:
            raise ValueError("Footprint checkpoint changed; use a new output directory")
        return report
    report = dict(
        site_id=row["site_id"],
        identity=identity,
        sources=signatures,
        frequency_bin=int(row["frequency_bin"]),
        split=row["split"],
        candidates=[],
        temporal_valid=0,
        common_support_valid=0,
        footprint_tile_valid=0,
    )
    try:
        offset = float(row["utc_offset_hours"])
        heights, height_rejects = metadata_events(paths["instrument"], "instrument", offset)
        canopy, canopy_rejects = metadata_events(paths["canopy"], "canopy", offset)
        report.update(
            instrument_events=heights,
            canopy_events=canopy,
            metadata_rejections=dict(instrument=height_rejects, canopy=canopy_rejects),
        )
        if not heights or not canopy:
            raise UnsupportedSource("No unambiguous dated instrument/aggregate canopy height")
        fluxnet = read_intervals(source, ["LE_F_MDS", "LE_F_MDS_QC"], offset)
        fluxes = read_intervals(paths["fluxes"], [*PARAMETERS.values(), "FOOTPRINT_FLAG"], offset)
        data = fluxnet.merge(
            fluxes, on=["TIMESTAMP_START", "TIMESTAMP_END"], how="left", validate="one_to_one"
        )
        starts, ends = data.TIMESTAMP_START.to_numpy(), data.TIMESTAMP_END.to_numpy()
        hc = history_values(canopy, starts, max_age_days=config["canopy_max_age_days"])
        zm = history_values(heights, starts) - 2 / 3 * hc
        params = {k: data[v].to_numpy(float) for k, v in PARAMETERS.items()}
        params["zm"] = zm
        physical = parameter_valid(canopy=hc, **params)
        # Flag 3 concerns NEE, not LE. Recomputed model validity still applies.
        provider = data.FOOTPRINT_FLAG.isin([0, 3]).to_numpy()
        good = physical & provider
        report["interval_counts"] = dict(
            total=len(data),
            finite_parameters=int(np.isfinite(np.array(list(params.values()))).all(axis=0).sum()),
            physical_valid=int(physical.sum()),
            provider_0_or_3=int(provider.sum()),
            joint_parameter_valid=int(good.sum()),
        )
        le, qc = data.LE_F_MDS.to_numpy(float), data.LE_F_MDS_QC.to_numpy(float)
        first = max(config["input_start"], config.get("start") or config["input_start"])
        last = min(
            config["as_of"][:10],
            config.get("end") or config["as_of"][:10],
            (pd.Timestamp(ends[-1], unit="s") + timedelta(days=1)).date().isoformat(),
        )
        first = max(first, pd.Timestamp(starts[0], unit="s").date().isoformat())
        rejects, labels = Counter(), []
        for start, end in windows(first, last, as_of=config["as_of"]):
            a, b = (
                np.searchsorted(ends, epoch(start), side="right"),
                np.searchsorted(starts, epoch(end)),
            )
            args = (starts[a:b], ends[a:b], le[a:b], qc[a:b], row["lat"], row["lon"], start, end)
            temporal = interval_et(*args)
            if not temporal["valid"]:
                rejects.update("temporal_" + r for r in temporal["reasons"])
                continue
            report["temporal_valid"] += 1
            common = interval_et(*args, support_valid=good[a:b])
            if not common["valid"]:
                rejects.update("common_" + r for r in common["reasons"])
                continue
            report["common_support_valid"] += 1
            use = np.flatnonzero(good[a:b] & np.isfinite(le[a:b]) & np.isin(qc[a:b], [0, 1])) + a
            durations = np.minimum(ends[use], epoch(end)) - np.maximum(starts[use], epoch(start))
            selected = [{k: float(v[i]) for k, v in params.items()} for i in use]
            mass, quality = window_mass(selected, durations, row["lon"], row["lat"])
            record = dict(
                target_time=start,
                target_end=end,
                temporal_et=temporal,
                et=common,
                **quality,
                supervision_ready=False,
                satellite_inputs_checked=False,
            )
            labels.append(record)
            if not quality["footprint_tile_valid"]:
                rejects["footprint_mass_or_quadrature"] += 1
                continue
            sid = fingerprint([identity, start])[:24]
            support_path = output / "footprints" / f"{sid}.zarr"
            if support_path.exists():
                support = zarr.open_group(str(support_path), mode="r")
                if support.attrs.get("identity") != identity or not support.attrs.get("complete"):
                    raise ValueError(f"Uncommitted footprint output: {support_path}")
            else:
                temporary = support_path.with_suffix(".partial.zarr")
                support = zarr.open_group(str(temporary), mode="w")
                for key, values in dict(
                    mass=mass,
                    weights=mass / mass.sum(),
                    interval_start=starts[use],
                    interval_end=ends[use],
                    interval_seconds=durations,
                ).items():
                    support.create_dataset(
                        key, data=values, compressor=Blosc(cname="zstd", clevel=3)
                    )
                support.attrs.update(
                    dict(
                        record,
                        identity=identity,
                        complete=True,
                        kind="label_support_only_not_training_sample",
                    )
                )
                temporary.rename(support_path)
            report["footprint_tile_valid"] += 1
            report["candidates"].append(
                dict(
                    candidate_id=sid,
                    site_id=row["site_id"],
                    physical_group_id=row["physical_group_id"],
                    spatial_group_id=row["spatial_group_id"],
                    split=row["split"],
                    frequency_bin=int(row["frequency_bin"]),
                    lon=row["lon"],
                    lat=row["lat"],
                    landcover=int(row["landcover"] or 0),
                    climate_class=int(row["climate_class"] or 0),
                    target_time=start,
                    target_end=end,
                    year=int(start[:4]),
                    month=int(start[5:7]),
                    et=common["value"],
                    captured_mass=quality["captured_mass"],
                    footprint_zarr=str(support_path),
                    label_record=str(output / "windows.json"),
                    record_index=len(labels) - 1,
                    satellite_inputs_checked=False,
                )
            )
        atomic_json(output / "windows.json", dict(identity=identity, windows=labels))
        report.update(status="complete", rejection_counts=dict(rejects))
    except UnsupportedSource as error:
        report.update(status="unsupported", reason=str(error))
    report["elapsed_seconds"] = monotonic() - began
    atomic_json(checkpoint, report)
    return report


def _process(task):
    return process_station(*task)


def prepare_icos_footprints(
    index,
    splits,
    strata,
    output,
    *,
    as_of,
    start=None,
    end=None,
    stations=None,
    workers=2,
    canopy_max_age_days=366,
):
    index, splits, strata, output = map(Path, (index, splits, strata, output))
    output.mkdir(parents=True, exist_ok=True)
    files = pd.read_parquet(index / "source_files.parquet")
    locations = pd.read_parquet(splits / "station_splits.parquet").merge(
        pd.read_parquet(strata / "station_strata.parquet")[
            ["location_id", "frequency_bin", "landcover", "climate_class"]
        ],
        on="location_id",
        validate="one_to_one",
    )
    files = files[files.source.eq("icos") & files.metadata_issue.eq("")].merge(
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
        validate="many_to_one",
    )
    files = files[files.frequency_bin.ge(0) & files.split.ne("unresolved")]
    if stations:
        wanted = {s if s.startswith("icos:") else "icos:" + s for s in stations}
        files = files[files.site_id.isin(wanted)]
        if set(files.site_id) != wanted:
            raise ValueError("Requested stations are missing or unresolved")
    if workers < 1 or canopy_max_age_days <= 0 or files.empty:
        raise ValueError("Positive resources/metadata age and eligible stations required")
    config = dict(
        version="site_icos_ffp_v1",
        as_of=as_of,
        start=start,
        end=end,
        stations=stations,
        input_start=json.loads((strata / "strata_report.json").read_text())["input_start"],
        canopy_max_age_days=canopy_max_age_days,
        displacement="2/3 * aggregate_canopy_height",
        roughness_sublayer="zm > 12.5 * .15 * canopy_height",
        ffp="Kljun_v1.42_umean",
        pblh_source="ICOS_FLUXES_provider_estimate_no_external_fill",
        time_basis="local_standard_to_UTC_using_SITEINFO_UTC_OFFSET",
        min_tile_mass=0.95,
        inputs_sha256={
            str(p): sha256(p)
            for p in (
                index / "source_files.parquet",
                splits / "station_splits.parquet",
                strata / "station_strata.parquet",
            )
        },
    )
    with locked(output / ".footprints.lock", blocking=False):
        path = output / "config.json"
        if path.exists() and json.loads(path.read_text()) != config:
            raise ValueError("Footprint definition changed; use a new output directory")
        atomic_json(path, config)
        reports = []
        columns = [
            "site_id",
            "source_path",
            "source_bytes",
            "source_mtime_ns",
            "metadata_sha256",
            "utc_offset_hours",
            "physical_group_id",
            "spatial_group_id",
            "split",
            "frequency_bin",
            "lon",
            "lat",
            "landcover",
            "climate_class",
        ]
        tasks = [(r, str(output), config) for r in files[columns].fillna(0).to_dict("records")]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for report in pool.map(_process, tasks):
                reports.append(report)
                print(
                    json.dumps(
                        {
                            k: report.get(k)
                            for k in (
                                "site_id",
                                "status",
                                "temporal_valid",
                                "common_support_valid",
                                "footprint_tile_valid",
                                "reason",
                                "elapsed_seconds",
                            )
                        }
                    ),
                    flush=True,
                )
                atomic_json(output / "progress.json", dict(done=len(reports), total=len(tasks)))
        candidates = [c for r in reports for c in r["candidates"]]
        pd.DataFrame(candidates).to_parquet(output / "candidates.parquet", index=False)
        summary = dict(
            complete=True,
            stations=len(reports),
            reports=reports,
            temporal_valid=sum(r["temporal_valid"] for r in reports),
            common_support_valid=sum(r["common_support_valid"] for r in reports),
            footprint_tile_valid=len(candidates),
            satellite_validated=0,
            scope="ICOS label/support candidates; satellite input pairing pending",
        )
        atomic_json(output / "report.json", summary)
        return summary
