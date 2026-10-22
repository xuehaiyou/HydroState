"""Storage primitives; task-specific schemas live with their data contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ArraySpec:
    tail_shape: tuple[int, ...]
    dtype: str


class ShardSchema(Protocol):
    """Structural interface accepted by the immutable shard writer."""

    version: str
    inputs: dict[str, ArraySpec]
    targets: dict[str, ArraySpec]
