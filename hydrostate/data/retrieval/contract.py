"""One strict contract for collection, training and inference; no legacy fallback."""

from dataclasses import dataclass
from datetime import date, timedelta

from hydrostate.storage.schema import ArraySpec

VERSION = "hydrostate_retrieval_5x8_v1"
PROFILE = "landsat_l1_toa_mean_v1"
OUTPUT_SCALE = 3
TIME_STEPS = 5
INPUT_SIZE = 120
OUTPUT_SIZE = INPUT_SIZE // OUTPUT_SCALE
SENSORS = {"s1": 2, "s2": 12, "landsat": 11}
QUALITY_CHANNELS = 4
COVER = ("water", "snow_ice", "land")
ET = ("et_water", "et_ice", "et_land")
TASKS = {
    **{name: dict(support="pixel", grid="output", station_index=None) for name in COVER},
    "soil_moisture": dict(support="tile_mean_weak", grid="scalar", station_index=None),
    **{name: dict(support="native", grid="native", station_index=None) for name in ET},
}


def intervals(start):
    t = date.fromisoformat(start)
    return [
        ((t + timedelta(days=d)).isoformat(), (t + timedelta(days=d + 8)).isoformat())
        for d in (-16, -8, 0, 8, 16)
    ]


def product_end(start):
    t = date.fromisoformat(start)
    return min(t + timedelta(days=8), date(t.year + 1, 1, 1)).isoformat()


@dataclass(frozen=True)
class RetrievalSchema:
    version: str = VERSION
    time_steps: int = TIME_STEPS
    height: int = INPUT_SIZE
    width: int = INPUT_SIZE
    native_capacity: int = 128

    @property
    def inputs(self):
        t, h, w = self.time_steps, self.height, self.width
        result = {}
        for name, channels in SENSORS.items():
            result[name] = ArraySpec((t, channels, h, w), "float16")
            result[f"{name}_valid"] = ArraySpec((t, 1, h, w), "uint8")
            # count, mean/min/max time offset (days from slot start).
            result[f"{name}_quality"] = ArraySpec((t, QUALITY_CHANNELS, h, w), "float16")
        result.update(
            sensor_valid=ArraySpec((t, len(SENSORS)), "uint8"),
            observation_time=ArraySpec((t, len(SENSORS)), "int64"),
            slot_bounds=ArraySpec((t, 2), "int64"),
            target_bounds=ArraySpec((2,), "int64"),
        )
        return result

    @property
    def targets(self):
        h, w = self.height // OUTPUT_SCALE, self.width // OUTPUT_SCALE
        specs = {}
        for name in COVER:
            specs[name] = ArraySpec((1, h, w), "float32")
            specs[f"{name}_valid"] = ArraySpec((1, h, w), "uint8")
        specs.update(
            cover_known=ArraySpec((1, h, w), "float32"),
            cover_days=ArraySpec((1, h, w), "float32"),
            cover_time_span=ArraySpec((1, h, w), "float32"),
            cover_observed_days=ArraySpec((1, h, w), "float32"),
            cover_unknown_fraction=ArraySpec((1, h, w), "float32"),
            soil_moisture=ArraySpec((1, 1, 1), "float32"),
            soil_moisture_valid=ArraySpec((1, 1, 1), "uint8"),
            soil_moisture_domain_valid=ArraySpec((1, h, w), "uint8"),
            soil_moisture_weight=ArraySpec((1, 1, 1), "float32"),
        )
        for name in ET:
            specs[f"{name}_native_value"] = ArraySpec((self.native_capacity,), "float32")
            specs[f"{name}_native_valid"] = ArraySpec((self.native_capacity,), "uint8")
            specs[f"{name}_native_weights"] = ArraySpec((self.native_capacity, h, w), "float32")
        return specs
