"""One strict contract for collection, training and inference; no legacy fallback."""

from dataclasses import dataclass
from datetime import date, timedelta

from hydrostate.storage.schema import ArraySpec

VERSION = "hydrostate_retrieval_5x8_native_v2"
PROFILE = "landsat_l1_toa_native_mean_v2"
# The existing centre-pool key is independent of the spectral storage version.
CANDIDATE_VERSION = "hydrostate_retrieval_5x8_v1"
INPUT_RESOLUTION = 10
OUTPUT_SCALE = 3
TIME_STEPS = 5
INPUT_SIZE = 120
OUTPUT_SIZE = INPUT_SIZE // OUTPUT_SCALE
SENSORS = {"s1": 2, "s2": 12, "landsat": 11}
QUALITY_CHANNELS = 4


@dataclass(frozen=True)
class InputGroup:
    sensor: str
    resolution: int
    bands: tuple[str, ...]

    def shape(self, height=INPUT_SIZE, width=INPUT_SIZE):
        if (
            height * INPUT_RESOLUTION % self.resolution
            or width * INPUT_RESOLUTION % self.resolution
        ):
            raise ValueError("Tile extent must contain whole native pixels")
        return (
            height * INPUT_RESOLUTION // self.resolution,
            width * INPUT_RESOLUTION // self.resolution,
        )

    def transform(self, reference):
        # Same outer bounds and CRS; scale affine pixel vectors, not the origin.
        ratio = self.resolution / INPUT_RESOLUTION
        return [v * ratio if i in (0, 1, 3, 4) else v for i, v in enumerate(reference)]


# Published product grids, not detector footprints (Landsat thermal is published at 30 m).
INPUT_GROUPS = {
    "s1_10m": InputGroup("s1", 10, ("VV", "VH")),
    "s2_10m": InputGroup("s2", 10, ("B2", "B3", "B4", "B8")),
    "s2_20m": InputGroup("s2", 20, ("B5", "B6", "B7", "B8A", "B11", "B12")),
    "s2_60m": InputGroup("s2", 60, ("B1", "B9")),
    "landsat_15m": InputGroup("landsat", 15, ("B8",)),
    "landsat_30m": InputGroup(
        "landsat", 30, ("B1", "B2", "B3", "B4", "B5", "B6", "B7", "B9", "B10", "B11")
    ),
}
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
        """Stored composites and exact QA/date support on each published band grid."""
        t = self.time_steps
        result = {}
        for name, group in INPUT_GROUPS.items():
            h, w = group.shape(self.height, self.width)
            result[name] = ArraySpec((t, len(group.bands), h, w), "float16")
            result[f"{name}_valid"] = ArraySpec((t, 1, h, w), "uint8")
            # count, mean/min/max time offset (days from slot start).
            result[f"{name}_quality"] = ArraySpec((t, QUALITY_CHANNELS, h, w), "float16")
        result.update(self.input_metadata)
        return result

    @property
    def input_metadata(self):
        t = self.time_steps
        return dict(
            sensor_valid=ArraySpec((t, len(SENSORS)), "uint8"),
            observation_time=ArraySpec((t, len(SENSORS)), "int64"),
            slot_bounds=ArraySpec((t, 2), "int64"),
            target_bounds=ArraySpec((2,), "int64"),
        )

    @property
    def model_inputs(self):
        """Locally aligned inputs; per-group dates remain distinct, never averaged."""
        t, h, w = self.time_steps, self.height, self.width
        result = dict(self.input_metadata)
        for sensor, channels in SENSORS.items():
            result[sensor] = ArraySpec((t, channels, h, w), "float16")
            result[f"{sensor}_valid"] = ArraySpec((t, 1, h, w), "uint8")
        for name in INPUT_GROUPS:
            result[f"{name}_valid"] = ArraySpec((t, 1, h, w), "uint8")
            result[f"{name}_quality"] = ArraySpec((t, QUALITY_CHANNELS, h, w), "float16")
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
