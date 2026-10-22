"""Per-worker bounded readers for immutable Zarr shards."""

from collections import OrderedDict
from typing import Any

import zarr


class ShardCache:
    """Small per-worker LRU cache of open Zarr groups."""

    def __init__(self, max_open: int = 8) -> None:
        self.max_open = max_open
        self._groups: OrderedDict[str, Any] = OrderedDict()

    def get(self, path: str) -> Any:
        if path in self._groups:
            self._groups.move_to_end(path)
            return self._groups[path]
        group = zarr.open_group(path, mode="r")
        self._groups[path] = group
        if len(self._groups) > self.max_open:
            _, evicted = self._groups.popitem(last=False)
            if hasattr(evicted.store, "close"):
                evicted.store.close()
        return group
