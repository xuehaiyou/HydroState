"""Atomic writer for fixed-size sampled Zarr shards."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import zarr
from numcodecs import Blosc

from .schema import ArraySpec, Schema


class SampleShardWriter:
    """Write one immutable input/label shard pair.

    Call ``write`` exactly ``num_samples`` times and then ``finalize``. Shards
    are first created with a ``.partial`` suffix and atomically renamed only
    after all arrays and metadata have been written.
    """

    def __init__(
        self,
        input_path: str,
        label_path: str,
        num_samples: int,
        schema: Schema | None = None,
        attrs: dict[str, Any] | None = None,
    ) -> None:
        self.schema = schema or Schema()
        self.num_samples = num_samples
        self.index = 0
        self.input_path = Path(input_path)
        self.label_path = Path(label_path)
        self.input_partial = Path(f"{input_path}.partial")
        self.label_partial = Path(f"{label_path}.partial")
        for path in (self.input_path, self.label_path, self.input_partial, self.label_partial):
            if path.exists():
                raise FileExistsError(path)
        self.input_partial.parent.mkdir(parents=True, exist_ok=True)
        self.label_partial.parent.mkdir(parents=True, exist_ok=True)
        compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
        self.input_group = zarr.open_group(str(self.input_partial), mode="w")
        self.label_group = zarr.open_group(str(self.label_partial), mode="w")
        self._create_arrays(self.input_group, self.schema.inputs, compressor)
        self._create_arrays(self.label_group, self.schema.targets, compressor)
        metadata = {"schema_version": self.schema.version, "num_samples": num_samples}
        metadata.update(attrs or {})
        self.input_group.attrs.update(metadata)
        self.label_group.attrs.update(metadata)

    def _create_arrays(self, group, specs: dict[str, ArraySpec], compressor) -> None:
        for name, spec in specs.items():
            group.create_dataset(
                name,
                shape=(self.num_samples, *spec.tail_shape),
                chunks=(1, *spec.tail_shape),
                dtype=spec.dtype,
                compressor=compressor,
                overwrite=False,
            )

    @staticmethod
    def _write_group(group, specs: dict[str, ArraySpec], index: int, values: dict) -> None:
        for name, spec in specs.items():
            if name not in values:
                raise KeyError(f"sample is missing required array {name!r}")
            value = np.asarray(values[name], dtype=spec.dtype)
            if value.shape != spec.tail_shape:
                raise ValueError(f"{name} has shape {value.shape}, expected {spec.tail_shape}")
            group[name][index] = value

    def write(self, inputs: dict[str, Any], targets: dict[str, Any]) -> int:
        if self.index >= self.num_samples:
            raise RuntimeError("shard is already full")
        row = self.index
        self._write_group(self.input_group, self.schema.inputs, row, inputs)
        self._write_group(self.label_group, self.schema.targets, row, targets)
        self.index += 1
        return row

    def finalize(self) -> None:
        if self.index != self.num_samples:
            raise RuntimeError(f"wrote {self.index} samples, expected {self.num_samples}")
        marker = {"complete": True, "num_samples": self.num_samples}
        for partial in (self.input_partial, self.label_partial):
            with (partial / "_SUCCESS.json").open("w", encoding="utf-8") as handle:
                json.dump(marker, handle)
        os.rename(self.input_partial, self.input_path)
        os.rename(self.label_partial, self.label_path)
