"""Compatibility exports for the shared training/regional time policies."""

from hydrostate.data.temporal_sampling import CENTERED, OFFSETS, PAST, context_offsets, policy
from hydrostate.data.temporal_sampling import observation_slots as regional_slots

__all__ = ["CENTERED", "OFFSETS", "PAST", "context_offsets", "policy", "regional_slots"]
