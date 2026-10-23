"""Release checks shared by the local CLI and Slurm preparation."""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

from hydrostate.data.gee.contracts import fingerprint
from hydrostate.data.spatial import geographic_split
from hydrostate.storage.packs import file_hash

from .alignment import grid_metadata
from .contract import PROFILE, VERSION, RetrievalSchema
from .labels import ice_centre_eligible
from .sampling import ICE_GROUP, SAMPLING_VERSION, SamplingLedger, inside_split_block, quotas


def audit(root, *, pixel_limit=64, require_complete=True, label_packs=None, label_specs=None):
    from .validation import validate_arrays

    root = Path(root)
    config = json.loads((root / "collection.json").read_text())
    if config.get("version") != VERSION or config.get("input_profile") != PROFILE:
        raise ValueError("Unsupported collection contract")
    if config.get("sampling_version") != SAMPLING_VERSION:
        raise ValueError("Unsupported sampling contract; require eight sampling groups")
    config_hash = fingerprint(
        {k: v for k, v in config.items() if k not in ("version", "input_profile")}
    )
    table = pd.read_parquet(root / "samples.parquet")
    if table.empty or table.sample_id.duplicated().any() or not (table.version == VERSION).all():
        raise ValueError("Invalid or empty manifest")
    counts = Counter(table.sampling_group)
    if require_complete:
        marker = json.loads((root / "_COLLECTION_SUCCESS.json").read_text())
        if (
            not marker.get("complete")
            or marker.get("version") != VERSION
            or marker.get("manifest_sha256") != file_hash(root / "samples.parquet")
            or counts != Counter(quotas(config["samples"]))
        ):
            raise ValueError("Collection has not completed all eight hard quotas")
    schema = RetrievalSchema()
    ledger = SamplingLedger(
        max_dates=config["max_dates_per_location"], min_days=config["min_date_gap_days"]
    )
    count = len(table) if pixel_limit == 0 else min(pixel_limit, len(table))
    selected = set(np.linspace(0, len(table) - 1, count, dtype=int))
    valid_counts = Counter()
    for index, row in enumerate(table.to_dict("records")):
        row["transform"] = json.loads(row["transform"])
        sampling_group, fb = row["sampling_group"], row["frequency_bin"]
        if (
            sampling_group not in range(8)
            or fb not in range(-1, 7)
            or (sampling_group != ICE_GROUP and fb != sampling_group)
            or (sampling_group == ICE_GROUP and row["candidate_target_time"] != row["target_time"])
            or bool(row["jrc_observed"]) != (fb >= 0)
        ):
            raise ValueError(f"Invalid sampling group/frequency/date: {row['sample_id']}")
        if (
            row["split"] != geographic_split(row, config["split_seed"])
            or not inside_split_block(row["lon"], row["lat"], config["split_margin_m"])
            or not ledger.allows(row, row["target_time"])
        ):
            raise ValueError(f"Invalid spatial split/overlap/date support: {row['sample_id']}")
        ledger.add(row)
        loaded = []
        for kind, specs in (("input", schema.inputs), ("label", schema.targets)):
            path = root / row[f"{kind}_path"]
            if not (path / "_SUCCESS.json").exists():
                raise ValueError(f"Incomplete shard: {path}")
            group = zarr.open_group(str(path), mode="r")
            if (
                group.attrs.get("schema_version") != VERSION
                or group.attrs.get("config_sha256") != config_hash
            ):
                raise ValueError(f"Shard contract mismatch: {path}")
            if kind == "input" and group.attrs.get("input_grids") != grid_metadata(row):
                raise ValueError(f"Native input grids disagree with tile bounds: {path}")
            for name, spec in specs.items():
                array = group[name]
                if array.shape != (1, *spec.tail_shape) or array.dtype != np.dtype(spec.dtype):
                    raise ValueError(f"Wrong shape/dtype: {path}/{name}")
            if kind == "label" and sampling_group == ICE_GROUP:
                centre = {name: group[name][0] for name in ("snow_ice", "snow_ice_valid")}
                if not ice_centre_eligible(centre, config["min_center_ice_fraction"]):
                    raise ValueError(f"Ice centre fails target-window coverage: {row['sample_id']}")
            if index in selected:
                loaded.append({name: group[name][0] for name in specs})
        if loaded:
            validate_arrays(*loaded)
            for name, value in loaded[1].items():
                if name.endswith("_valid"):
                    valid_counts[name] += int(value.sum())
    result = dict(
        samples=len(table),
        version=VERSION,
        sampling_group_counts=dict(counts),
        frequency_counts=dict(Counter(table.frequency_bin)),
        unique_locations=len(ledger.dates),
        pixel_audited=len(selected),
        audited_valid_counts=dict(valid_counts),
        manifest_sha256=file_hash(root / "samples.parquet"),
    )
    if label_packs or label_specs:
        from .label_packs import audit_label_packs

        result["label_extensions"] = audit_label_packs(
            root / "samples.parquet", label_packs, label_specs, root=root, pixel_limit=pixel_limit
        )
    return result
