"""Station observations share native satellite inputs, not product label supports."""

from dataclasses import dataclass

from hydrostate.data.retrieval.contract import RetrievalSchema
from hydrostate.storage.schema import ArraySpec

VERSION = "hydrostate_site_5x8_am_v1"
OBSERVATIONS = ("site_sm", "site_et")


@dataclass(frozen=True)
class SiteSchema(RetrievalSchema):
    version: str = VERSION
    observation_capacity: int = 1

    @property
    def targets(self):
        k, h, w = self.observation_capacity, self.height // 3, self.width // 3
        if k < 1:
            raise ValueError("Station shard requires a positive observation capacity")
        result = {}
        for name in OBSERVATIONS:
            result.update(
                {
                    f"{name}_native_value": ArraySpec((k,), "float32"),
                    f"{name}_native_valid": ArraySpec((k,), "uint8"),
                    f"{name}_native_weights": ArraySpec((k, h, w), "float32"),
                }
            )
        return result
