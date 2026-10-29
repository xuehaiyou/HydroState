"""Bounded, per-worker caches for immutable channel-first Zarr spatial arrays."""

from collections import OrderedDict

import numpy as np

from hydrostate.storage.reader import ShardCache


class SpatialChunkCache:
    """Keep array metadata and decoded chunks, with an explicit retained-byte limit.

    Returned windows own their memory. Cached chunks are read-only; callers cannot
    modify a later sample through an earlier window. Only C×H×W arrays are accepted.
    """

    def __init__(self, max_bytes=512 * 1024**2, max_arrays=384):
        if max_bytes < 0 or max_arrays < 1:
            raise ValueError("Invalid spatial cache budget")
        self.max_bytes, self.max_arrays = int(max_bytes), int(max_arrays)
        self.bytes = 0
        self.groups = ShardCache(max_open=max_arrays)
        self.arrays, self.chunks = OrderedDict(), OrderedDict()

    def _array(self, path, name):
        key = (path, name)
        if key not in self.arrays:
            array = self.groups.get(path)[name]
            if array.ndim != 3 or array.chunks[0] != array.shape[0]:
                raise ValueError("Spatial cache requires whole-channel C×H×W chunks")
            self.arrays[key] = array
            if len(self.arrays) > self.max_arrays:
                self.arrays.popitem(last=False)
        self.arrays.move_to_end(key)
        return self.arrays[key]

    def _chunk(self, path, name, array, y, x):
        key = (path, name, y, x)
        if key in self.chunks:
            self.chunks.move_to_end(key)
            return self.chunks[key]
        _, height, width = array.chunks
        value = array[:, y:y+height, x:x+width]
        value.flags.writeable = False
        if value.nbytes <= self.max_bytes:
            while self.bytes + value.nbytes > self.max_bytes:
                _, evicted = self.chunks.popitem(last=False)
                self.bytes -= evicted.nbytes
            self.chunks[key] = value
            self.bytes += value.nbytes
        return value

    def read(self, path, name, rows, cols):
        path = str(path)
        array = self._array(path, name)
        if not self.max_bytes:
            return array[:, rows, cols]
        y0, y1, ys = rows.indices(array.shape[1])
        x0, x1, xs = cols.indices(array.shape[2])
        if ys != 1 or xs != 1:
            raise ValueError("Spatial cache requires contiguous windows")
        result = np.empty((array.shape[0], max(0, y1-y0), max(0, x1-x0)), array.dtype)
        _, height, width = array.chunks
        for y in range(y0 // height * height, y1, height):
            for x in range(x0 // width * width, x1, width):
                chunk = self._chunk(path, name, array, y, x)
                a, b = max(y0, y), max(x0, x)
                c, d = min(y1, y+chunk.shape[1]), min(x1, x+chunk.shape[2])
                result[:, a-y0:c-y0, b-x0:d-x0] = chunk[:, a-y:c-y, b-x:d-x]
        return result
