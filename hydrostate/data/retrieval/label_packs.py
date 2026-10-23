"""Independent, immutable labels joined to existing imagery by sample identity."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from hydrostate.storage.packs import VariablePack, VariablePackWriter, file_hash, validate_samples
from hydrostate.storage.reader import ShardCache

from .contract import OUTPUT_SIZE, PROFILE, TASKS, VERSION, intervals, product_end
from .specs import resolve_label_specs


def period_bounds(row):
    start = str(row["target_time"])
    end = row.get("target_end", product_end(start))

    def epoch(value):
        return int(pd.Timestamp(value, tz="UTC").timestamp())

    slots = intervals(start)
    return np.array([epoch(start), epoch(end)]), np.array([epoch(slots[0][0]), epoch(slots[-1][1])])


def validate_observation(name, spec, values, target_bounds, input_bounds, *, native=False):
    prefix = f"{name}_native" if native and spec.grid == "native" else name
    value = values[f"{prefix}_value"] if prefix != name else values[name]
    valid = values[f"{prefix}_valid"].astype(bool)
    if value.shape != valid.shape:
        raise ValueError(f"{name}: value and mask shapes differ")
    if not np.isfinite(value[valid]).all():
        raise ValueError(f"{name}: nonfinite valid label")
    if spec.valid_min is not None and np.any(value[valid] < spec.valid_min):
        raise ValueError(f"{name}: label below declared physical range")
    if spec.valid_max is not None and np.any(value[valid] > spec.valid_max):
        raise ValueError(f"{name}: label above declared physical range")
    time = np.asarray(values[f"{name}_time"])
    if time.shape != (2,) or not np.issubdtype(time.dtype, np.integer) or time[1] < time[0]:
        raise ValueError(f"{name}: invalid label time bounds")
    if valid.any():
        if spec.temporal == "target_interval" and not np.array_equal(time, target_bounds):
            raise ValueError(f"{name}: label must match the target interval exactly")
        if spec.temporal == "instant" and time[0] != time[1]:
            raise ValueError(f"{name}: instant label must have equal time bounds")
        if spec.temporal != "instant" and time[0] >= time[1]:
            raise ValueError(f"{name}: interval label must have positive duration")
        if time[0] < input_bounds[0] or time[1] > input_bounds[1]:
            raise ValueError(f"{name}: label time is outside the input window")
    if spec.grid == "native":
        weights = values[f"{prefix}_weights"]
        if (
            not np.isfinite(weights).all()
            or (weights < 0).any()
            or not np.allclose(weights[valid].sum(axis=(-2, -1)), 1, atol=1e-5)
        ):
            raise ValueError(f"{name}: invalid native area weights")


class RetrievalLabelPackWriter(VariablePackWriter):
    """Values are already QA-filtered and in declared physical units; no input downloads."""

    def __init__(self, path, manifest, *, label_specs, source, version, shard_size=256):
        if not source:
            raise ValueError("Label pack source provenance is required")
        self.definitions = {
            name: spec
            for name, spec in resolve_label_specs(label_specs).items()
            if name in label_specs
        }
        if set(self.definitions) & set(TASKS):
            raise ValueError("Extension packs cannot replace built-in coupled labels")
        table = pd.read_parquet(manifest)
        validate_samples(table)
        if not (table.version == VERSION).all() or not (table.input_profile == PROFILE).all():
            raise ValueError("Label packs require the current input collection")
        self.bounds = {row["sample_id"]: period_bounds(row) for row in table.to_dict("records")}
        super().__init__(
            path,
            manifest,
            kind="label",
            version=version,
            shard_size=shard_size,
            source=dict(
                producer="hydrostate_retrieval_label_pack", input_version=VERSION, provenance=source
            ),
            variables={
                name: spec.pack_variable((OUTPUT_SIZE, OUTPUT_SIZE))
                for name, spec in self.definitions.items()
            },
        )

    def write(self, sample_id, values, *, provenance=None):
        if sample_id not in self.bounds:
            raise ValueError(f"Unknown sample_id: {sample_id}")
        for name, spec in self.definitions.items():
            validate_observation(name, spec, values, *self.bounds[sample_id])
        return super().write(sample_id, values, provenance=provenance)


class LabelPackSet:
    def __init__(self, options, samples, root, definitions):
        self.definitions, self.packs, self.identity = definitions, [], []
        selected = set()
        for raw in options or []:
            option = dict(raw)
            path = Path(option.pop("path"))
            path = path if path.is_absolute() else Path(root) / path
            pack = VariablePack(path, samples, kind="label", include_times=True, **option)
            for name, variable in pack.variables.items():
                if name in TASKS or name in selected:
                    raise ValueError(f"Multiple or built-in label sources for {name}")
                if name not in definitions:
                    raise ValueError(f"No label specification registered for {name}")
                expected = definitions[name].pack_variable((OUTPUT_SIZE, OUTPUT_SIZE))
                if any(variable.get(k) != v for k, v in expected.items()):
                    raise ValueError(f"Pack units/grid/time/specification mismatch: {name}")
                selected.add(name)
            self.packs.append(pack)
            self.identity.append(
                dict(
                    path=str(raw["path"]),
                    variables=list(pack.variables),
                    metadata_sha256=file_hash(pack.path / "metadata.json"),
                    index_sha256=pack.metadata["index_sha256"],
                    max_uncertainty=pack.max_uncertainty,
                )
            )
        if missing := set(definitions) - set(TASKS) - selected:
            raise ValueError(f"Missing label packs for registered tasks: {sorted(missing)}")

    def read(self, sample_id, cache, target_bounds, input_bounds):
        labels, times = {}, {}
        for pack in self.packs:
            values = pack.read(sample_id, cache)
            for name in pack.variables:
                spec = self.definitions[name]
                validate_observation(name, spec, values, target_bounds, input_bounds, native=True)
                mask = values[f"{name}_native_valid" if spec.grid == "native" else f"{name}_valid"]
                times[name] = values.pop(f"{name}_time") if mask.any() else target_bounds.copy()
                values.pop(f"{name}_time", None)
            labels.update(values)
        return labels, times


def audit_label_packs(manifest, options, label_specs, *, root=None, pixel_limit=64):
    table = pd.read_parquet(manifest)
    packs = LabelPackSet(
        options, table, root or Path(manifest).parent, resolve_label_specs(label_specs)
    )
    cache, counts = ShardCache(), {}
    # Inspect each shard's schema, including shards beyond the pixel sample.
    for pack in packs.packs:
        shards = {}
        for shard, row in pack.locations.values():
            shards[shard] = max(row, shards.get(shard, 0))
        for shard, row in shards.items():
            group = cache.get(shard)
            for name, spec in pack.variables.items():
                shapes = {
                    name: (spec["shape"], spec["dtype"]),
                    f"{name}_valid": (spec["shape"], "bool"),
                    f"{name}_time": ([2], "int64"),
                }
                if spec["grid"] == "native":
                    shapes[f"{name}_weights"] = (
                        [*spec["shape"], *spec["spatial_shape"]],
                        "float32",
                    )
                for key, (shape, dtype) in shapes.items():
                    array = group[key]
                    if (
                        array.shape[1:] != tuple(shape)
                        or row >= array.shape[0]
                        or array.dtype != np.dtype(dtype)
                    ):
                        raise ValueError(f"Invalid label pack array: {shard}/{key}")
    count = len(table) if pixel_limit == 0 else min(pixel_limit, len(table))
    for i in np.linspace(0, len(table) - 1, count, dtype=int):
        row = table.iloc[i]
        values, _ = packs.read(row.sample_id, cache, *period_bounds(row))
        for key, value in values.items():
            if key.endswith("_valid"):
                counts[key] = counts.get(key, 0) + int(value.sum())
    return dict(packs=packs.identity, audited_samples=count, valid_counts=counts)


def build_label_pack(manifest, records, specification, output, shard_size=256):
    records = Path(records)
    table = pd.read_parquet(records)
    if not {"sample_id", "npz_path"}.issubset(table) or table.sample_id.duplicated().any():
        raise ValueError("Records require unique sample_id and npz_path")
    with RetrievalLabelPackWriter(
        output, manifest, shard_size=shard_size, **specification
    ) as writer:
        for row in table.itertuples():
            path = Path(row.npz_path)
            if not path.is_absolute():
                path = records.parent / path
            with np.load(path, allow_pickle=False) as values:
                writer.write(row.sample_id, values)
        return writer.finalize()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--records", required=True, type=Path)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--shard-size", type=int, default=256)
    args = parser.parse_args()
    print(
        build_label_pack(
            args.manifest,
            args.records,
            json.loads(args.spec.read_text()),
            args.output,
            args.shard_size,
        )
    )
