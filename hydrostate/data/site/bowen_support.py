"""Rematch existing ICOS footprint evidence to corrected half-hour supports."""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import zarr

from hydrostate.data.gee.contracts import fingerprint

from .bowen_audit import prepare
from .footprint import window_mass
from .icos_footprints import PARAMETERS, history_values, read_intervals
from .temporal import epoch, interval_et


@lru_cache(maxsize=4)
def corrected_intervals(path, offset):
    return prepare(path, offset)[0]


@lru_cache(maxsize=4)
def footprint_parameters(report_path, offset, max_age_days):
    report = json.loads(Path(report_path).read_text())
    frame = read_intervals(report["sources"]["fluxes"]["path"], list(PARAMETERS.values()), offset)
    starts = frame.TIMESTAMP_START.to_numpy()
    hc = history_values(report["canopy_events"], starts, max_age_days=max_age_days)
    zm = history_values(report["instrument_events"], starts) - 2 / 3 * hc
    parameters = {k: frame[v].to_numpy(float) for k, v in PARAMETERS.items()}
    parameters["zm"] = zm
    return starts, parameters


def rematch_footprint(row, meta, evidence, output):
    """Reuse mass only when every original contributing interval remains valid.

    Otherwise recalculate FFP for the retained intervals using the original measured
    footprint parameters. Never substitute uniform weights for a failed footprint.
    """
    old = zarr.open_group(evidence["footprint_zarr"], mode="r")
    starts, ends = np.asarray(old["interval_start"]), np.asarray(old["interval_end"])
    data = corrected_intervals(meta["source_path"], float(meta["utc_offset_hours"]))
    i = np.searchsorted(data["starts"], starts)
    matched = i < len(data["starts"])
    i = np.minimum(i, len(data["starts"]) - 1)
    matched &= (data["starts"][i] == starts) & (data["ends"][i] == ends)
    if not matched.all():
        raise ValueError("Footprint and corrected flux supports do not align exactly")
    keep = data["corrected_mask"][i]
    args = dict(
        latitude=row["lat"],
        longitude=row["lon"],
        start=row["target_time"],
        end=row["target_end"],
        min_valid=0,
        min_day_night=0.75,
        min_measured=0.5,
        support_valid=keep,
    )
    corrected = interval_et(starts, ends, data["corrected"][i], data["qc"][i], **args)
    raw = interval_et(starts, ends, data["raw"][i], data["qc"][i], **args)
    if not corrected["valid"]:
        return None, dict(reason="corrected_footprint_time_coverage", quality=corrected)
    quality = dict(old.attrs)
    if keep.all():
        mass = np.asarray(old["mass"])
    else:
        report_path = str(Path(evidence["footprint_zarr"]).parent.parent / "report.json")
        old_config = json.loads(
            (Path(evidence["footprint_zarr"]).parents[3] / "config.json").read_text()
        )
        times, parameters = footprint_parameters(
            report_path, float(meta["utc_offset_hours"]), old_config["canopy_max_age_days"]
        )
        j = np.searchsorted(times, starts[keep])
        if np.any(j >= len(times)) or not np.array_equal(times[j], starts[keep]):
            raise ValueError("Missing original FFP parameters for retained intervals")
        selected = [{k: float(v[n]) for k, v in parameters.items()} for n in j]
        duration = np.minimum(ends[keep], epoch(row["target_end"])) - np.maximum(
            starts[keep], epoch(row["target_time"])
        )
        mass, quality = window_mass(selected, duration, row["lon"], row["lat"])
    if not quality["footprint_tile_valid"]:
        return None, dict(reason="corrected_footprint_mass_or_quadrature", quality=quality)
    path = Path(output) / "footprints" / (row["candidate_id"] + ".zarr")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        group = zarr.open_group(str(path), mode="w")
        for key, value in dict(
            mass=mass,
            weights=mass / mass.sum(),
            interval_start=starts[keep],
            interval_end=ends[keep],
        ).items():
            group.create_dataset(key, data=value)
        group.attrs.update(
            dict(
                quality,
                complete=True,
                identity=fingerprint([row["candidate_id"], "bowen_75_50"]),
                et=corrected,
                et_uncorrected=raw,
            )
        )
    return dict(
        path=str(path),
        et=corrected,
        et_uncorrected=raw,
        temporal_basis="corrected_valid_intersection_with_existing_FFP",
    ), None
