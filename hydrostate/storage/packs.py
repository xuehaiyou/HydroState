"""Immutable, sample-ID-indexed input/label packs, independent of legacy shards."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from numcodecs import Blosc

IDENTITY_COLUMNS = (
    "sample_id", "split", "crs", "transform", "target_time", "tile_id", "site_id",
    "height", "width", "resolution", "time_steps", "sample_key",
    "target_end", "version", "input_profile",
)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sample_fingerprints(table):
    """Bind extensions to existing identities, independent of paths and label contents."""
    columns = [key for key in IDENTITY_COLUMNS if key in table]
    records = json.loads(table[columns].to_json(orient="records", date_format="iso"))
    return {
        row["sample_id"]: hashlib.sha256(
            json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        for row in records
    }


def validate_samples(table):
    if not {"sample_id", "split"}.issubset(table):
        raise ValueError("Sample manifest requires sample_id and split")
    ids = table.sample_id
    if ids.isna().any() or ids.duplicated().any() or not ids.map(
        lambda x: isinstance(x, str) and bool(x)
    ).all():
        raise ValueError("sample_id must contain unique, nonempty strings")


def validate_variables(kind, variables):
    if kind not in {"input", "label"} or not variables:
        raise ValueError("A pack requires kind=input/label and nonempty variables")
    result = {}
    for name, raw in variables.items():
        if (not re.fullmatch(r"[a-z][a-z0-9_]*", name)
                or name.endswith(("_valid", "_time"))
                or name.startswith(("site_", "et_native", "prediction_"))
                or name in {"sensor_valid", "observation_time", "metainfo"}):
            raise ValueError(f"Reserved or invalid variable name: {name}")
        spec = dict(raw)
        shape = tuple(spec.get("shape", ()))
        if not shape or any(type(n) is not int or n <= 0 for n in shape):
            raise ValueError(f"{name}: shape must contain positive integers")
        if not isinstance(spec.get("units"), str) or not spec["units"]:
            raise ValueError(f"{name}: explicit units are required (use '1' for dimensionless)")
        dtype = np.dtype(spec.get("dtype", "float32"))
        if dtype not in (np.dtype("float16"), np.dtype("float32")):
            raise ValueError(f"{name}: values must be float16 or float32")
        if kind == "input":
            if name in {"s1", "s2", "landsat"}:
                raise ValueError("Satellite replacement requires its normalization adapter")
            temporal = spec.get("temporal")
            if temporal not in {"static", "sequence"}:
                raise ValueError(f"{name}: temporal must be static or sequence")
            if len(shape) != (3 if temporal == "static" else 4) or spec.get("grid") != "input":
                raise ValueError(f"{name}: use CHW/TCHW aligned to the sample input grid")
        elif spec.get("grid") == "native":
            spatial = spec.get("spatial_shape", [])
            if (len(shape) != 1 or len(spatial) != 2
                    or any(type(n) is not int or n <= 0 for n in spatial)):
                raise ValueError(
                    f"{name}: native labels require shape=[capacity], spatial_shape=[H,W]")
        elif len(shape) != 3 or shape[0] != 1 or spec.get("grid") not in {
            "input", "output", "scalar"
        }:
            raise ValueError(f"{name}: label shape must be 1HW with an explicit grid")
        if kind == "label" and spec["grid"] == "scalar" and shape != (1, 1, 1):
            raise ValueError(f"{name}: scalar labels must have shape [1, 1, 1]")
        if "uncertainty_units" in spec and (kind != "label"
                or not isinstance(spec["uncertainty_units"], str)
                or not spec["uncertainty_units"]):
            raise ValueError(f"{name}: uncertainty requires a label and explicit units")
        if "default_max_uncertainty" in spec and (
                "uncertainty_units" not in spec
                or not isinstance(spec["default_max_uncertainty"], (int, float))
                or not np.isfinite(spec["default_max_uncertainty"])
                or spec["default_max_uncertainty"] <= 0):
            raise ValueError(f"{name}: invalid default uncertainty threshold")
        if "time_support" in spec and (kind != "label" or spec["time_support"] not in {
                "instant", "interval"}):
            raise ValueError(f"{name}: time_support must be instant or interval for labels")
        result[name] = dict(spec, shape=list(shape), dtype=str(dtype))
    return result


class VariablePackWriter:
    """Stream aligned arrays into a new version; publish only on successful finalize.

    Values and equally shaped boolean masks are mandatory. Sequence inputs also
    require ``<name>_time``: UTC epoch seconds per observation. Producers perform
    reprojection, temporal selection and physical unit conversion before writing.
    """

    def __init__(self, path, manifest, *, kind, variables, source, version, shard_size=256,
                 chunk_samples=1):
        self.path = Path(path)
        if self.path.exists():
            raise FileExistsError(self.path)
        if not source or not version or type(shard_size) is not int or shard_size <= 0:
            raise ValueError("source, version and a positive integer shard_size are required")
        self.variables = validate_variables(kind, variables)
        if (type(chunk_samples) is not int or chunk_samples <= 0
                or shard_size % chunk_samples):
            raise ValueError("chunk_samples must be a positive divisor of shard_size")
        self.chunk_samples, self.pending = chunk_samples, []
        self.table = pd.read_parquet(manifest)
        validate_samples(self.table)
        self.identities = sample_fingerprints(self.table)
        self.metadata = dict(format_version=1, kind=kind, version=str(version), source=source,
                             variables=self.variables)
        self.shard_size = shard_size
        self.partial = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex}.partial")
        self.partial.mkdir(parents=True)
        self.rows, self.seen = [], set()
        self.group = None
        self.closed = False

    def _new_shard(self):
        self.shard = f"data/part-{len(self.rows) // self.shard_size:05d}.zarr"
        self.group = zarr.open_group(str(self.partial / self.shard), mode="w")
        compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
        for name, spec in self.variables.items():
            shape = tuple(spec["shape"])
            for key, dtype in ((name, spec["dtype"]), (f"{name}_valid", "bool")):
                self.group.create_dataset(key, shape=(self.shard_size, *shape),
                                          chunks=(self.chunk_samples, *shape),
                                          dtype=dtype, compressor=compressor)
            if spec.get("grid") == "native":
                weights = (*shape, *spec["spatial_shape"])
                self.group.create_dataset(f"{name}_weights", shape=(self.shard_size, *weights),
                                          chunks=(self.chunk_samples, *weights), dtype="float32",
                                          compressor=compressor)
            if "uncertainty_units" in spec:
                self.group.create_dataset(f"{name}_uncertainty", shape=(self.shard_size, *shape),
                                          chunks=(self.chunk_samples, *shape), dtype="float32",
                                          compressor=compressor)
            if spec.get("temporal") == "sequence":
                self.group.create_dataset(f"{name}_time", shape=(self.shard_size, shape[0]),
                                          chunks=(self.chunk_samples, shape[0]), dtype="int64")
            if "time_support" in spec:
                self.group.create_dataset(f"{name}_time", shape=(self.shard_size, 2),
                                          chunks=(self.chunk_samples, 2), dtype="int64")

    def write(self, sample_id, values, *, provenance=None):
        if self.closed:
            raise RuntimeError("Pack writer is closed")
        if sample_id not in self.identities or sample_id in self.seen:
            raise ValueError(f"Unknown or duplicate sample_id: {sample_id}")
        prepared = {}
        for name, spec in self.variables.items():
            data = np.asarray(values[name], dtype=spec["dtype"])
            raw_valid = np.asarray(values[f"{name}_valid"])
            if data.shape != tuple(spec["shape"]) or raw_valid.shape != data.shape:
                raise ValueError(f"{name}: value/mask shapes must both equal {spec['shape']}")
            if not np.isin(raw_valid, [0, 1]).all():
                raise ValueError(f"{name}: masks must be boolean or 0/1")
            valid = raw_valid.astype(bool)
            if not np.isfinite(data[valid]).all():
                raise ValueError(f"{name}: nonfinite value marked valid")
            prepared[name] = np.where(valid, data, 0)
            prepared[f"{name}_valid"] = valid
            if "time_support" in spec:
                bounds = np.asarray(values[f"{name}_time"])
                if (bounds.shape != (2,) or not np.issubdtype(bounds.dtype, np.integer)
                        or (valid.any() and bounds[0] <= 0) or bounds[1] < bounds[0]
                        or (spec["time_support"] == "instant" and bounds[0] != bounds[1])):
                    raise ValueError(f"{name}: invalid observation time bounds")
                prepared[f"{name}_time"] = bounds.astype("int64")
            if "uncertainty_units" in spec:
                uncertainty = np.asarray(values[f"{name}_uncertainty"], dtype="float32")
                if (uncertainty.shape != data.shape or not np.isfinite(uncertainty[valid]).all()
                        or (uncertainty[valid] < 0).any()):
                    raise ValueError(f"{name}: uncertainty must be finite and nonnegative")
                prepared[f"{name}_uncertainty"] = np.where(valid, uncertainty, 0)
            if spec.get("grid") == "native":
                weights = np.asarray(values[f"{name}_weights"], dtype="float32")
                if (weights.shape != (*data.shape, *spec["spatial_shape"])
                        or not np.isfinite(weights).all() or (weights < 0).any()):
                    raise ValueError(f"{name}: invalid native area weights")
                if not np.allclose(weights[valid].sum(axis=(-2, -1)), 1, atol=1e-5):
                    raise ValueError(f"{name}: each valid native footprint must sum to one")
                prepared[f"{name}_weights"] = np.where(valid[:, None, None], weights, 0)
            if spec.get("temporal") == "sequence":
                times = np.asarray(values[f"{name}_time"])
                if times.shape != (data.shape[0],) or not np.issubdtype(times.dtype, np.integer):
                    raise ValueError(f"{name}: time must be an integer vector of UTC epoch seconds")
                if np.any(times[valid.reshape(data.shape[0], -1).any(1)] <= 0):
                    raise ValueError(f"{name}: valid observations require positive timestamps")
                prepared[f"{name}_time"] = times.astype("int64")
        row = len(self.rows) % self.shard_size
        if row == 0:
            self._new_shard()
        self.pending.append(prepared)
        entry = dict(sample_id=sample_id, shard=self.shard, row=row,
                     sample_fingerprint=self.identities[sample_id])
        if provenance is not None:
            entry["provenance"] = json.dumps(provenance, sort_keys=True, allow_nan=False)
        self.rows.append(entry)
        self.seen.add(sample_id)
        if len(self.pending) == self.chunk_samples:
            self._flush()

    def _flush(self):
        if not self.pending:
            return
        end = (len(self.rows) - 1) % self.shard_size + 1
        start = end - len(self.pending)
        for key in self.pending[0]:
            self.group[key][start:end] = np.stack([row[key] for row in self.pending])
        self.pending.clear()

    def finalize(self):
        if self.closed or not self.rows:
            raise RuntimeError("Cannot publish an empty or closed pack")
        self._flush()
        index = self.partial / "index.parquet"
        pd.DataFrame(self.rows).to_parquet(index, index=False)
        self.metadata["index_sha256"] = file_hash(index)
        self.metadata["num_samples"] = len(self.rows)
        (self.partial / "metadata.json").write_text(
            json.dumps(self.metadata, indent=2, sort_keys=True) + "\n"
        )
        (self.partial / "_SUCCESS.json").write_text(json.dumps({
            "metadata_sha256": file_hash(self.partial / "metadata.json")
        }) + "\n")
        # Serialize final publication across writers targeting the same version.
        lock = self.path.with_name(f".{self.path.name}.publish.lock")
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            if self.path.exists():
                raise FileExistsError(self.path)
            os.rename(self.partial, self.path)
            self.closed = True
        finally:
            os.close(descriptor)
            lock.unlink()
        return self.path

    def abort(self):
        if not self.closed:
            shutil.rmtree(self.partial)
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        if not self.closed:
            self.abort()


class VariablePack:
    """Validate one frozen index, then resolve selected variables without directory scans."""

    def __init__(self, path, samples, *, kind, variables=None, max_uncertainty=None,
                 include_times=False):
        self.path = Path(path).resolve()
        marker = json.loads((self.path / "_SUCCESS.json").read_text())
        self.metadata = json.loads((self.path / "metadata.json").read_text())
        if marker["metadata_sha256"] != file_hash(self.path / "metadata.json"):
            raise ValueError(f"Pack metadata changed after publication: {self.path}")
        if self.metadata.get("format_version") != 1 or self.metadata.get("kind") != kind:
            raise ValueError(f"Wrong pack format/kind: {self.path}")
        if file_hash(self.path / "index.parquet") != self.metadata["index_sha256"]:
            raise ValueError(f"Pack index changed after publication: {self.path}")
        available = validate_variables(kind, self.metadata["variables"])
        selected = list(available) if variables is None else list(variables)
        if (not selected or len(set(selected)) != len(selected)
                or set(selected) - available.keys()):
            raise ValueError(f"Invalid selected variables: {selected}")
        self.variables = {key: available[key] for key in selected}
        self.max_uncertainty = {
            name: spec["default_max_uncertainty"] for name, spec in self.variables.items()
            if "default_max_uncertainty" in spec
        } | dict(max_uncertainty or {})
        for name, threshold in self.max_uncertainty.items():
            if (name not in self.variables or "uncertainty_units" not in self.variables[name]
                    or not isinstance(threshold, (int, float))
                    or not np.isfinite(threshold) or threshold <= 0):
                raise ValueError(f"Invalid uncertainty filter: {name}={threshold}")
        index = pd.read_parquet(self.path / "index.parquet")
        if not {"sample_id", "shard", "row", "sample_fingerprint"}.issubset(index):
            raise ValueError("Pack index is missing required columns")
        if index.sample_id.isna().any() or index.sample_id.duplicated().any():
            raise ValueError("Pack index must contain unique nonempty sample IDs")
        if len(index) != self.metadata["num_samples"]:
            raise ValueError("Pack index count differs from metadata")
        identities = sample_fingerprints(samples)
        expected = index.sample_id.map(identities)
        if expected.isna().any() or not (expected == index.sample_fingerprint).all():
            raise ValueError(
                "Pack samples differ from the sample manifest (identity/split/grid/time)")
        if not pd.api.types.is_integer_dtype(index.row) or (index.row < 0).any():
            raise ValueError("Pack row offsets must be nonnegative integers")
        if index.duplicated(["shard", "row"]).any():
            raise ValueError("Pack samples cannot share a physical row")
        self.locations = {}
        self.legacy_times = {}
        source = self.metadata.get("source", {})
        legacy_gee = (include_times and kind == "label" and isinstance(source, dict)
                      and str(source.get("producer", "")).startswith("gee_ecosystem_labels"))
        # Resolve each shard once; WSL/network metadata calls per sample are costly.
        shards = {name: (self.path / name).resolve() for name in index.shard.unique()}
        for row in index.itertuples():
            shard = shards[row.shard]
            if not shard.is_relative_to(self.path):
                raise ValueError("Pack shards must be inside the pack directory")
            self.locations[row.sample_id] = (str(shard), int(row.row))
            if legacy_gee:
                from hydrostate.data.target_times import product_time_bounds

                scenes = json.loads(row.provenance)["scenes"]
                self.legacy_times[row.sample_id] = {
                    name: product_time_bounds(scenes, name) for name in self.variables
                    if "time_support" not in self.variables[name]}

    def read(self, sample_id, cache, *, read_values=True):
        location = self.locations.get(sample_id)
        group, row = (None, None) if location is None else (cache.get(location[0]), location[1])
        result = {}
        for name, spec in self.variables.items():
            shape = tuple(spec["shape"])
            if group is None:
                value, valid = np.zeros(shape, dtype=spec["dtype"]), np.zeros(shape, dtype=bool)
            else:
                value = (np.asarray(group[name][row]) if read_values else
                         np.zeros(shape, dtype=spec['dtype']))
                valid = np.asarray(group[f"{name}_valid"][row], dtype=bool)
                if value.shape != shape or valid.shape != shape:
                    raise ValueError(f"Pack array shape differs from metadata: {name}")
            result[name], result[f"{name}_valid"] = value, valid
            if "time_support" in spec:
                bounds = (np.zeros(2, dtype="int64") if group is None
                          else np.asarray(group[f"{name}_time"][row]))
                if (bounds.shape != (2,) or not np.issubdtype(bounds.dtype, np.integer)
                        or (valid.any() and bounds[0] <= 0) or bounds[1] < bounds[0]
                        or (spec["time_support"] == "instant" and bounds[0] != bounds[1])):
                    raise ValueError(f"{name}: invalid stored observation time bounds")
                result[f"{name}_time"] = bounds
            elif name in self.legacy_times.get(sample_id, {}):
                result[f"{name}_time"] = self.legacy_times[sample_id][name]
            if "uncertainty_units" in spec:
                uncertainty = (np.zeros(shape, dtype="float32") if group is None
                               else np.asarray(group[f"{name}_uncertainty"][row]))
                if uncertainty.shape != shape:
                    raise ValueError(f"Pack uncertainty differs from metadata: {name}")
                result[f"{name}_uncertainty"] = uncertainty
                if name in self.max_uncertainty:
                    valid = valid & (uncertainty <= self.max_uncertainty[name])
                    result[f"{name}_valid"] = valid
            if spec.get("grid") == "native":
                weights = (np.zeros((*shape, *spec["spatial_shape"]), dtype="float32")
                           if group is None else np.asarray(group[f"{name}_weights"][row]))
                if weights.shape != (*shape, *spec["spatial_shape"]):
                    raise ValueError(f"Pack native weights differ from metadata: {name}")
                # Trim padding before collation; every sample still has at least one row.
                selected = np.flatnonzero(valid) if valid.any() else np.array([0])
                result.pop(name)
                result.pop(f"{name}_valid")
                result[f"{name}_native_value"] = value[selected]
                result[f"{name}_native_valid"] = valid[selected]
                result[f"{name}_native_weights"] = weights[selected]
                if "uncertainty_units" in spec:
                    result[f"{name}_native_uncertainty"] = result.pop(
                        f"{name}_uncertainty")[selected]
            if spec.get("temporal") == "sequence":
                result[f"{name}_time"] = (
                    np.zeros(shape[0], dtype="int64") if group is None
                    else np.asarray(group[f"{name}_time"][row])
                )
        return result
