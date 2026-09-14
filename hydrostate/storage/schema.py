"""Canonical sampled-window storage contract."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ArraySpec:
    tail_shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class Schema:
    time_steps: int = 4
    height: int = 128
    width: int = 128
    version: str = "1.0"

    @property
    def inputs(self) -> dict[str, ArraySpec]:
        t, h, w = self.time_steps, self.height, self.width
        return {
            "s1": ArraySpec((t, 2, h, w), "float16"),
            "s2": ArraySpec((t, 12, h, w), "float16"),
            "landsat": ArraySpec((t, 11, h, w), "float16"),
            "sensor_valid": ArraySpec((t, 3), "uint8"),
            "observation_time": ArraySpec((t, 3), "int64"),
            "s1_valid": ArraySpec((t, 1, h, w), "uint8"),
            "s2_valid": ArraySpec((t, 1, h, w), "uint8"),
            "landsat_valid": ArraySpec((t, 1, h, w), "uint8"),
        }

    @property
    def targets(self) -> dict[str, ArraySpec]:
        h, w = self.height, self.width
        specs: dict[str, ArraySpec] = {}
        for task in ("water", "soil_moisture", "et"):
            specs[task] = ArraySpec((1, h, w), "float32")
            specs[f"{task}_valid"] = ArraySpec((1, h, w), "uint8")
        specs.update(
            {
                "site_value": ArraySpec((3,), "float32"),
                "site_valid": ArraySpec((3,), "uint8"),
                "site_footprint": ArraySpec((1, h, w), "float32"),
            }
        )
        return specs


INPUT_ARRAYS = tuple(Schema().inputs)
TARGET_ARRAYS = tuple(Schema().targets)
