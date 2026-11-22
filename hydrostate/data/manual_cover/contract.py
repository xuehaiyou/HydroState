"""Explicit pure-point assumption and raster area fractions on the retrieval grid."""

from dataclasses import dataclass
from pathlib import Path

from hydrostate.data.gee.contracts import fingerprint
from hydrostate.data.retrieval.contract import RetrievalSchema
from hydrostate.storage.packs import file_hash
from hydrostate.storage.schema import ArraySpec

VERSION = "hydrostate_manual_cover_v2"
CLASSES = ("water", "snow_ice", "land")
FULL_AREA_TOLERANCE = 1e-5


def producer_identity():
    from hydrostate.data.retrieval.plan import data_signature

    folder = Path(__file__).parent
    return fingerprint(
        dict(
            inputs=data_signature(folder.parents[2]),
            files={p.name: file_hash(p) for p in sorted(folder.glob("*.py"))},
        )
    )


@dataclass(frozen=True)
class ManualCoverSchema(RetrievalSchema):
    version: str = VERSION

    @property
    def targets(self):
        specs = {}
        for name in CLASSES:
            specs[name] = ArraySpec((1, 40, 40), "float32")
            specs[name + "_valid"] = ArraySpec((1, 40, 40), "uint8")
        specs.update(
            valid_area_fraction=ArraySpec((1, 40, 40), "float32"),
            point_count=ArraySpec((1, 40, 40), "int32"),
            point_conflict=ArraySpec((1, 40, 40), "uint8"),
        )
        return specs
