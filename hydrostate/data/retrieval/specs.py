"""Canonical product, input and label specifications; no network or model imports."""

from dataclasses import asdict, dataclass
from math import isfinite

from hydrostate.storage.schema import ArraySpec


@dataclass(frozen=True)
class Source:
    collection: str
    start: str | None = None
    end: str | None = None


SOURCES = {
    "s1": (Source("COPERNICUS/S1_GRD"),),
    "s2": (Source("COPERNICUS/S2_SR_HARMONIZED"),),
    "landsat": (Source("LANDSAT/LC08/C02/T1"), Source("LANDSAT/LC09/C02/T1")),
    "dw": (Source("GOOGLE/DYNAMICWORLD/V1"),),
    "pml": (Source("projects/pml_evapotranspiration/PML/OUTPUT/PML_V22a"),),
    "sm": (
        Source("NASA/SMAP/SPL3SMP_E/005", "2015-01-01", "2023-12-04"),
        Source("NASA/SMAP/SPL3SMP_E/006", "2023-12-04", "2100-01-01"),
    ),
}


@dataclass(frozen=True)
class BandSpec:
    name: str
    resolution_m: int
    units: str
    normalization_key: str = ""
    offset: float = 0
    divisor: float = 1


@dataclass(frozen=True)
class InputSpec:
    source: str
    modality: str
    bands: tuple[BandSpec, ...]
    qa: str
    normalization: str
    aggregation: str = "daily_mean"
    temporal: str = "five_8day_slots"

    def __post_init__(self):
        if (
            self.source not in SOURCES
            or not self.bands
            or len({b.name for b in self.bands}) != len(self.bands)
        ):
            raise ValueError("Inputs require a known source and unique, nonempty bands")
        if self.temporal != "five_8day_slots" or self.aggregation not in (
            "daily_mean",
            "linear_power_daily_mean",
        ):
            raise ValueError("Input temporal/aggregation policy requires a producer adapter")
        if any(b.resolution_m <= 0 or not b.units or b.divisor <= 0 for b in self.bands):
            raise ValueError("Invalid band resolution, units or normalization")


INPUT_SPECS = {
    "s1": InputSpec(
        "s1",
        "sentinel1",
        tuple(BandSpec(b, 10, "dB", b.lower()) for b in ("VV", "VH")),
        "grd_iw_valid",
        "pretrained",
        "linear_power_daily_mean",
    ),
    "s2": InputSpec(
        "s2",
        "sentinel2_l2a",
        tuple(
            BandSpec(b, r, "reflectance_DN_1e-4", "B" + b[1:].zfill(2) if b != "B8A" else b)
            for b, r in (
                ("B2", 10),
                ("B3", 10),
                ("B4", 10),
                ("B8", 10),
                ("B5", 20),
                ("B6", 20),
                ("B7", 20),
                ("B8A", 20),
                ("B11", 20),
                ("B12", 20),
                ("B1", 60),
                ("B9", 60),
            )
        ),
        "scl_clear_water_snow",
        "pretrained",
    ),
    "landsat": InputSpec(
        "landsat",
        "landsat",
        tuple(
            BandSpec(
                b,
                15 if b == "B8" else 30,
                "K" if b in ("B10", "B11") else "1",
                offset=290 if b in ("B10", "B11") else 0,
                divisor=80 if b in ("B10", "B11") else 2,
            )
            for b in ("B8", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B9", "B10", "B11")
        ),
        "landsat_c2_clear_water_snow",
        "toa_physical",
    ),
}


@dataclass(frozen=True)
class LabelSpec:
    source: str
    band: str
    units: str
    support: str
    resolution_m: float | str
    qa: str
    aggregation: str
    temporal: str = "target_interval"
    activation: str = "identity"
    scale: float = 1
    offset: float = 0
    valid_min: float | None = None
    valid_max: float | None = None
    coverage_class: str | None = None
    capacity: int = 128

    def __post_init__(self):
        if self.support not in ("pixel", "native", "window_mean", "tile_mean_weak"):
            raise ValueError(f"Unsupported label support: {self.support}")
        if self.temporal not in ("target_interval", "instant", "interval"):
            raise ValueError(f"Unsupported label temporal support: {self.temporal}")
        if not all((self.source, self.band, self.units, self.qa, self.aggregation)):
            raise ValueError("Labels require source, band, units, QA and aggregation")
        if self.activation not in ("identity", "sigmoid", "softplus") or self.capacity < 1:
            raise ValueError("Invalid activation or native capacity")
        if type(self.capacity) is not int or not all(
            isfinite(v) for v in (self.scale, self.offset)
        ):
            raise ValueError("Native capacity must be an integer; conversion must be finite")
        if any(v is not None and not isfinite(v) for v in (self.valid_min, self.valid_max)):
            raise ValueError("Physical bounds must be finite")
        if (
            self.valid_min is not None
            and self.valid_max is not None
            and self.valid_min > self.valid_max
        ):
            raise ValueError("Physical bounds are reversed")
        if not isinstance(self.resolution_m, str) and (
            not isfinite(self.resolution_m) or self.resolution_m <= 0
        ):
            raise ValueError("Product resolution must be positive")

    @property
    def grid(self):
        return (
            "output"
            if self.support == "pixel"
            else "native"
            if self.support == "native"
            else "scalar"
        )

    def task(self):
        return dict(support=self.support, grid=self.grid, station_index=None)

    def arrays(self, name, shape, capacity=None):
        if self.grid == "native":
            n = self.capacity if capacity is None else capacity
            return {
                f"{name}_native_value": ArraySpec((n,), "float32"),
                f"{name}_native_valid": ArraySpec((n,), "uint8"),
                f"{name}_native_weights": ArraySpec((n, *shape), "float32"),
            }
        dimensions = (1, *shape) if self.grid == "output" else (1, 1, 1)
        return {
            name: ArraySpec(dimensions, "float32"),
            f"{name}_valid": ArraySpec(dimensions, "uint8"),
        }

    def pack_variable(self, shape):
        dimensions = (
            [self.capacity]
            if self.grid == "native"
            else [1, *shape]
            if self.grid == "output"
            else [1, 1, 1]
        )
        spec = dict(
            shape=dimensions,
            dtype="float32",
            units=self.units,
            grid=self.grid,
            time_support="instant" if self.temporal == "instant" else "interval",
            retrieval_spec=asdict(self),
        )
        if self.grid == "native":
            spec["spatial_shape"] = list(shape)
        return spec


LABEL_SPECS = {
    name: LabelSpec(
        "dw",
        band,
        "1",
        "pixel",
        30,
        "confident_cover",
        "daily_mean_area_mean",
        valid_min=0,
        valid_max=1,
        coverage_class=name,
    )
    for name, band in (("water", "water"), ("snow_ice", "snow_and_ice"), ("land", "land_classes"))
}
LABEL_SPECS["soil_moisture"] = LabelSpec(
    "sm",
    "soil_moisture_am",
    "m3/m3",
    "tile_mean_weak",
    "published_grid_and_33km_screen",
    "am_recommended_land_dominated",
    "daily_am_mean",
    activation="sigmoid",
    valid_min=0,
    valid_max=1,
)
LABEL_SPECS.update(
    {
        name: LabelSpec(
            "pml",
            "ET" if cls == "land" else "Ew",
            "mm/day",
            "native",
            "published_grid",
            "nonnegative_pure_cover",
            "product_period",
            activation="softplus",
            scale=0.01,
            valid_min=0,
            coverage_class=cls,
        )
        for name, cls in (("et_water", "water"), ("et_ice", "snow_ice"), ("et_land", "land"))
    }
)


def resolve_label_specs(extra=None):
    """Extensions are explicit per configuration, never process-global mutations."""
    result = dict(LABEL_SPECS)
    for name, raw in (extra or {}).items():
        spec = raw if isinstance(raw, LabelSpec) else LabelSpec(**raw)
        if name in result and spec != result[name]:
            raise ValueError(f"Cannot redefine built-in label: {name}")
        if name not in result and spec.support == "tile_mean_weak":
            raise ValueError("tile_mean_weak is reserved for the SM land coupling")
        result[name] = spec
    # The common task validator also rejects reserved variable names.
    from hydrostate.tasks import task_config

    task_config({name: spec.task() for name, spec in result.items()})
    return result


def specification_snapshot(labels=None):
    return dict(
        inputs={name: asdict(spec) for name, spec in INPUT_SPECS.items()},
        sources={name: [asdict(s) for s in values] for name, values in SOURCES.items()},
        labels={name: asdict(spec) for name, spec in (labels or LABEL_SPECS).items()},
    )
