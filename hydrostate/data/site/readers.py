"""Read extracted source files once; never open the observation database.

ISMN headers and FLUXNET BIF names follow Site_Process's existing adapters. The
subdaily records are retained here because its daily importer cannot recover mornings.
"""

import csv
import re
import shlex
from pathlib import Path

import numpy as np
import pandas as pd

from .archives import sha256
from .temporal import epoch, interval_et, morning_sm, windows


class UnsupportedSource(ValueError):
    """Known source with insufficient metadata/quality support for the declared target."""


def ismn_header(path):
    with Path(path).open() as stream:
        fields = shlex.split(stream.readline())
    if len(fields) != 9 or "_sm_" not in Path(path).name:
        raise UnsupportedSource("Not an ISMN separate-header soil-moisture file")
    database, network, site, lat, lon, elevation, lower, upper, sensor = fields
    lower, upper = float(lower), float(upper)
    if not 0 <= lower <= upper <= 0.05:
        raise UnsupportedSource("Depth outside the declared shallow 0–5 cm target")
    if any(
        word in sensor.lower() for word in ("cosmic", "hydroinnova", "gps", "gnss")
    ) or network.upper() in {"COSMOS", "PBO_H2O"}:
        raise UnsupportedSource("Non-point soil-moisture support requires its own operator")
    lat, lon = float(lat), float(lon)
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise UnsupportedSource("Invalid ISMN coordinates")
    return dict(
        source="ismn",
        database=database,
        network=network,
        site_id=f"ismn:{network}:{site}",
        group_id=f"ismn:{network}:{site}",
        lat=lat,
        lon=lon,
        sensor_id=Path(path).stem,
        depth_from_m=lower,
        depth_to_m=upper,
        instrument=sensor,
        support_type="point" if lower == upper else "point_layer",
        time_basis="UTC",
        unit="m3 m-3",
    )


def _unique_times(frame, columns):
    if frame["utc"].duplicated().any():
        for _, records in frame[frame.utc.duplicated(keep=False)].groupby("utc"):
            if len(records[columns].drop_duplicates()) > 1:
                raise UnsupportedSource("Conflicting duplicate timestamps require source review")
        frame = frame.drop_duplicates("utc")
    return frame.sort_values("utc").reset_index(drop=True)


def ismn_windows(path, start, end, as_of):
    meta = ismn_header(path)
    frame = pd.read_csv(
        path,
        sep=r"\s+",
        skiprows=1,
        header=None,
        names=["date", "time", "value", "qc", "provider_qc"],
    )
    stamps = pd.to_datetime(frame.date + " " + frame.time, format="%Y/%m/%d %H:%M", utc=True)
    frame["utc"] = stamps.dt.as_unit("s").astype("int64")
    frame = _unique_times(frame, ["value", "qc", "provider_qc"])
    times, values = frame.utc.to_numpy(), frame.value.to_numpy()
    quality = frame.qc.eq("G").to_numpy()
    provenance = dict(path=str(Path(path).resolve()), sha256=sha256(path))
    result = []
    for left, right in windows(start, end, as_of=as_of):
        a, b = np.searchsorted(times, [epoch(left), epoch(right)])
        label = morning_sm(times[a:b], values[a:b], quality[a:b], meta["lon"], left, right)
        result.append(
            dict(
                meta,
                target_time=left,
                target_end=right,
                source_file=provenance,
                sm=label,
                et=None,
                supervision_ready=label["valid"],
            )
        )
    return result


def fluxnet_metadata(path):
    """Fail closed on unresolved relocation/offset changes, instead of applying the last row."""
    path = Path(path)
    bifs = list(path.parent.glob("*_BIF_*.csv"))
    if len(bifs) != 1:
        raise UnsupportedSource("One unambiguous FLUXNET BIF file is required")
    with bifs[0].open(encoding="utf-8-sig", newline="") as stream:
        records = list(csv.DictReader(stream))

    def one(name, required=True):
        values = {
            r["DATAVALUE"]
            for r in records
            if r["VARIABLE"] == name and r["DATAVALUE"] not in ("", "-9999")
        }
        if not values and not required:
            return "unknown"
        if len(values) != 1:
            raise UnsupportedSource(f"Unresolved historical {name}: {sorted(values)}")
        return values.pop()

    sites = {r["SITE_ID"] for r in records}
    if len(sites) != 1:
        raise UnsupportedSource("BIF contains multiple stations")
    site = sites.pop()
    lat, lon, offset = (
        float(one("LOCATION_LAT")),
        float(one("LOCATION_LONG")),
        float(one("UTC_OFFSET")),
    )
    if not (-90 <= lat <= 90 and -180 <= lon <= 180 and -14 <= offset <= 14):
        raise UnsupportedSource("Invalid coordinates or UTC offset")
    return dict(
        source="fluxnet",
        site_id=f"flux:{site}",
        group_id=f"flux:{site}",
        lat=lat,
        lon=lon,
        utc_offset_hours=offset,
        time_basis="local_standard",
        landcover=one("IGBP", False),
        metadata_file=str(bifs[0]),
        metadata_sha256=sha256(bifs[0]),
        support_type="footprint_unresolved",
    )


def fluxnet_windows(path, start, end, as_of):
    if not re.search(r"_FLUXNET_(?:FLUXMET_|FULLSET_)?(?:HH|HR)_", Path(path).name):
        raise UnsupportedSource("Expected FLUXNET HH/HR, never daily QC or ERA5")
    meta = fluxnet_metadata(path)
    columns = pd.read_csv(path, nrows=0).columns
    required = ["TIMESTAMP_START", "TIMESTAMP_END", "LE_F_MDS", "LE_F_MDS_QC"]
    if not set(required).issubset(columns):
        raise UnsupportedSource("Missing standard LE or source-specific HH/HR QC")
    chosen = required + (["LE_CORR"] if "LE_CORR" in columns else [])
    frame = pd.read_csv(
        path,
        usecols=chosen,
        na_values=[-9999],
        dtype={"TIMESTAMP_START": str, "TIMESTAMP_END": str},
    )
    for source, target in (("TIMESTAMP_START", "utc"), ("TIMESTAMP_END", "end_utc")):
        local = pd.to_datetime(frame[source], format="%Y%m%d%H%M", utc=True)
        frame[target] = local.dt.as_unit("s").astype("int64") - int(meta["utc_offset_hours"] * 3600)
    frame = _unique_times(frame, ["end_utc", *chosen[2:]])
    times, ends = frame.utc.to_numpy(), frame.end_utc.to_numpy()
    if np.any(ends[:-1] > times[1:]) or np.any(ends <= times) or np.any(ends - times > 3600):
        raise UnsupportedSource("Non-HH/HR or overlapping flux intervals")
    provenance = dict(path=str(Path(path).resolve()), sha256=sha256(path))
    result = []
    for left, right in windows(start, end, as_of=as_of):
        a, b = (
            np.searchsorted(ends, epoch(left), side="right"),
            np.searchsorted(times, epoch(right)),
        )
        subset = frame.iloc[a:b]
        args = (times[a:b], ends[a:b])
        kwargs = dict(latitude=meta["lat"], longitude=meta["lon"], start=left, end=right)
        label = interval_et(*args, subset.LE_F_MDS, subset.LE_F_MDS_QC, **kwargs)
        corrected = (
            interval_et(*args, subset.LE_CORR, subset.LE_F_MDS_QC, **kwargs)
            if "LE_CORR" in subset
            else None
        )
        if corrected:
            corrected["quality_basis"] = "LE_F_MDS_QC; no independent closure-quality guarantee"
        result.append(
            dict(
                meta,
                target_time=left,
                target_end=right,
                source_file=provenance,
                sm=None,
                et=label,
                et_corrected=corrected,
                supervision_ready=False,
                support_rejection="Footprint or documented spatial proxy required",
            )
        )
    return result
