"""Area-weighted support for complete source pixels inside an existing sample."""

from __future__ import annotations

import json

import numpy as np
import shapely
from affine import Affine
from pyproj import Transformer


def _project(geometry, transformer):
    coordinates = shapely.get_coordinates(geometry)
    x, y = transformer.transform(coordinates[:, 0], coordinates[:, 1])
    return shapely.set_coordinates(geometry.copy(), np.column_stack((x, y)))


class NativeSupport:
    """Cache geometry across products sharing a projection within one sample.

    Products are fetched with nearest-neighbour sampling on the input grid. A
    complete native cell is sampled at its centre; its target is never inferred
    from a partially intersecting cell. Area weights use a local equal-area CRS.
    """

    def __init__(self, sample, shape=(120, 120), factor=3):
        self.sample, self.shape, self.factor = sample, shape, factor
        if any(n % factor for n in shape):
            raise ValueError("Input shape must be divisible by output factor")
        self.output_shape = tuple(n // factor for n in shape)
        self.grid = Affine(*sample["transform"])
        self.area_crs = (f"+proj=laea +lat_0={sample['lat']} +lon_0={sample['lon']} "
                         "+datum=WGS84 +units=m +type=crs")
        self.cache = {}
        self.pixels = None

    @staticmethod
    def _rectangles(affine, x, y, size):
        x, y = np.asarray(x), np.asarray(y)
        corners = np.stack((np.stack((x, y), -1), np.stack((x + size, y), -1),
                            np.stack((x + size, y + size), -1),
                            np.stack((x, y + size), -1)), axis=-2)
        a = np.array([[affine.a, affine.b], [affine.d, affine.e]])
        return shapely.polygons(corners @ a.T + [affine.c, affine.f])

    def geometry(self, projection):
        key = json.dumps(projection, sort_keys=True)
        if key in self.cache:
            return self.cache[key]
        h, w = self.shape
        to_area = Transformer.from_crs(self.sample["crs"], self.area_crs,
                                       always_xy=True, force_over=True)
        if self.pixels is None:
            yy, xx = np.mgrid[0:h:self.factor, 0:w:self.factor]
            pixels = self._rectangles(self.grid, xx.ravel(), yy.ravel(), self.factor)
            self.pixels = _project(shapely.segmentize(pixels, 7.5), to_area)
            self.tree = shapely.STRtree(self.pixels)
            outline = self._rectangles(self.grid, np.array([0]), np.array([0]), w)
            if h != w:
                raise ValueError("Native labels currently require square sample windows")
            self.outline = shapely.segmentize(outline, 7.5)
            self.footprint = _project(self.outline, to_area)[0]
        native = Affine(*projection["transform"])
        # NASA MODLAND sphere: https://modis-land.gsfc.nasa.gov/GCTP.html
        native_crs = ("+proj=sinu +R=6371007.181 +units=m +no_defs +type=crs"
                      if projection["crs"] == "SR-ORG:6974" else projection["crs"])
        to_native = Transformer.from_crs(self.sample["crs"], native_crs,
                                         always_xy=True, force_over=True)
        boundary = shapely.get_coordinates(self.outline)
        nx, ny = to_native.transform(boundary[:, 0], boundary[:, 1])
        cx, cy = ~native * (np.asarray(nx), np.asarray(ny))
        low = np.floor([cx.min(), cy.min()]).astype(int)
        high = np.ceil([cx.max(), cy.max()]).astype(int)
        yy, xx = np.mgrid[low[1]:high[1], low[0]:high[0]]
        xx, yy = xx.ravel(), yy.ravel()
        cells = self._rectangles(native, xx, yy, 1)
        native_area = Transformer.from_crs(native_crs, self.area_crs,
                                           always_xy=True, force_over=True)
        cells = _project(shapely.segmentize(cells, max(abs(native.a), abs(native.e)) / 16),
                         native_area)
        to_grid = Transformer.from_crs(native_crs, self.sample["crs"],
                                       always_xy=True, force_over=True)
        gx, gy = ~self.grid * to_grid.transform(*(native * (xx + .5, yy + .5)))
        cols, rows = np.floor(gx).astype(int), np.floor(gy).astype(int)
        weights, centres = [], []
        for cell, row, col in zip(cells, rows, cols):
            if (cell.area <= 0 or cell.difference(self.footprint).area > cell.area * 1e-7
                    or not (0 <= row < h and 0 <= col < w)):
                continue
            indices = self.tree.query(cell, predicate="intersects")
            area = shapely.area(shapely.intersection(self.pixels[indices], cell))
            if not np.isclose(area.sum(), cell.area, rtol=1e-6):
                raise ValueError("Area weights do not cover the complete native pixel")
            overlap = np.zeros(np.prod(self.output_shape), np.float32)
            overlap[indices] = area / area.sum()
            weights.append(overlap.reshape(self.output_shape))
            centres.append((row, col))
        result = (np.asarray(weights, np.float32).reshape(-1, *self.output_shape), centres)
        self.cache[key] = result
        return result

    def targets(self, values, valid, projection, capacity=64):
        if values.shape != self.shape or valid.shape != self.shape:
            raise ValueError("Source raster differs from the sample input grid")
        result = dict(value=np.zeros(capacity, np.float32), valid=np.zeros(capacity, bool),
                      weights=np.zeros((capacity, *self.output_shape), np.float32))
        if not valid.any():
            return result
        weights, centres = self.geometry(projection)
        selected = [i for i, (row, col) in enumerate(centres) if valid[row, col]]
        if len(selected) > capacity:
            raise ValueError(f"{len(selected)} native pixels exceed capacity={capacity}")
        for j, i in enumerate(selected):
            row, col = centres[i]
            result["value"][j] = values[row, col]
            result["valid"][j] = True
            result["weights"][j] = weights[i]
        return result
