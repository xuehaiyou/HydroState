"""Point labels are classes at locations, never fractional-area ground truth."""

from dataclasses import dataclass
from pathlib import Path

from hydrostate.data.gee.contracts import fingerprint
from hydrostate.data.retrieval.contract import RetrievalSchema
from hydrostate.storage.schema import ArraySpec

VERSION = "hydrostate_cover_points_v1"
CLASSES = ("water", "snow_ice", "land")
CLASS_MAP = {1: 1, 2: 1, 3: 1, 4: 2, 5: 0}


def config_identity(config):
    return fingerprint({k: v for k, v in config.items() if k not in ("version", "input_profile")})


@dataclass(frozen=True)
class CoverPointSchema(RetrievalSchema):
    version: str = VERSION
    observation_capacity: int = 1

    @property
    def targets(self):
        k = self.observation_capacity
        if k < 1:
            raise ValueError("Point capacity must be positive")
        return {
            "cover_point_class": ArraySpec((k,), "int8"),
            "cover_point_original_class": ArraySpec((k,), "int8"),
            "cover_point_valid": ArraySpec((k,), "uint8"),
            "cover_point_rc": ArraySpec((k, 2), "int16"),
            "cover_point_lonlat": ArraySpec((k, 2), "float64"),
            "cover_point_date": ArraySpec((k,), "int64"),
            "cover_point_index": ArraySpec((k,), "int64"),
        }


def producer_identity():
    """Bind pixel-producing dependencies and the independent point implementation."""
    from hydrostate.data.retrieval.plan import data_signature
    from hydrostate.storage.packs import file_hash

    repo = Path(__file__).resolve().parents[3]
    files = ("contract.py", "prepare.py", "collection.py")
    return fingerprint(
        dict(
            inputs=data_signature(repo),
            points={name: file_hash(Path(__file__).with_name(name)) for name in files},
        )
    )
