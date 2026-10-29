import numpy as np
import pytest
import zarr

from hydrostate.storage.spatial_cache import SpatialChunkCache


@pytest.mark.parametrize("budget", [0, 900, 4096, 1024**2])
def test_cached_windows_match_original_across_chunks_edges_and_evictions(tmp_path, budget):
    values = np.arange(3*19*23, dtype="f4").reshape(3, 19, 23)
    group = zarr.open_group(str(tmp_path / "input.zarr"), mode="w")
    group.create_dataset("values", data=values, chunks=(3, 8, 9))
    cache = SpatialChunkCache(max_bytes=budget)
    windows = [(slice(0, 6), slice(3, 11)), (slice(7, 19), slice(8, 23)),
               (slice(0, 19), slice(0, 23)), (slice(0, 6), slice(3, 11))]
    for rows, cols in windows:
        actual = cache.read(tmp_path / "input.zarr", "values", rows, cols)
        np.testing.assert_array_equal(actual, values[:, rows, cols])
        actual[:] = -1
        np.testing.assert_array_equal(cache.read(tmp_path / "input.zarr", "values", rows, cols),
                                      values[:, rows, cols])
        assert cache.bytes <= budget
        assert cache.bytes == sum(value.nbytes for value in cache.chunks.values())


def test_repeated_window_avoids_store_reads(tmp_path, monkeypatch):
    group = zarr.open_group(str(tmp_path / "input.zarr"), mode="w")
    group.create_dataset("valid", data=np.ones((1, 16, 16), "u1"), chunks=(1, 8, 8))
    cache = SpatialChunkCache(max_bytes=1024)
    expected = cache.read(tmp_path / "input.zarr", "valid", slice(3, 13), slice(4, 14))
    def fail(*args, **kwargs):
        raise AssertionError("Repeated window unexpectedly accessed Zarr")
    monkeypatch.setattr(zarr.Array, "__getitem__", fail)
    actual = cache.read(tmp_path / "input.zarr", "valid", slice(3, 13), slice(4, 14))
    np.testing.assert_array_equal(actual, expected)
