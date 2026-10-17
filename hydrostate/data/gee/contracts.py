"""Stable request identities, grid transforms and atomic local metadata."""

import fcntl
import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path

from affine import Affine


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def atomic_json(path, value):
    tmp = Path(str(path) + ".part")
    with tmp.open("w") as handle:
        json.dump(value, handle, sort_keys=True, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


@contextmanager
def locked(path, *, blocking=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        flags = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
        fcntl.flock(handle, flags)
        yield


def input_transform(grid):
    halo = grid["halo"] * 30
    return Affine(10, 0, grid["left"] - halo, 0, -10, grid["top"] + halo)


def input_shape(grid):
    halo = grid["halo"] * 6
    return grid["height"] * 3 + halo, grid["width"] * 3 + halo


def map_point(affine, col, row):
    """Compatible with both stable Affine 2.x and the 3.x matrix-operator API."""
    return (affine.a * col + affine.b * row + affine.c, affine.d * col + affine.e * row + affine.f)
