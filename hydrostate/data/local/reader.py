"""Windowed raster reads on the common grid, with bounded per-thread GDAL handles."""

from collections import OrderedDict
from concurrent.futures import Future
from pathlib import Path
from threading import local

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window

from hydrostate.data.gee.contracts import input_shape, input_transform
from hydrostate.data.gee.products import BANDS
from hydrostate.data.input_products import LANDSAT_SR_BANDS, harmonize_s2, landsat_l2_pixels
from hydrostate.storage.quality_cache import QualityCache


class RasterReader:
    def __init__(self, grid, scenes=()):
        self.grid = grid
        self.shape = input_shape(grid)
        self.local = local()
        self.scenes = {scene['asset_id']: scene for scene in scenes}

    def band(self, scene, name, y, x, height, width):
        if not hasattr(self.local, 'handles'):
            self.local.handles, self.local.verified = OrderedDict(), set()
        if scene['asset_id'] not in self.local.verified:
            for filename, expected in scene['snapshot'].items():
                stat = Path(filename).stat()
                if [stat.st_size, stat.st_mtime_ns] != expected:
                    raise ValueError(f'Indexed source changed: {filename}; use a new root')
            self.local.verified.add(scene['asset_id'])
        path = scene['files'][name]
        handles = self.local.handles
        if path not in handles:
            src = rasterio.open(path)
            try:
                vrt = WarpedVRT(src, crs=self.grid['crs'], transform=input_transform(self.grid),
                                width=self.shape[1], height=self.shape[0],
                                dtype='float32', nodata=float('nan'),
                                resampling=Resampling.nearest, warp_mem_limit=32)
            except BaseException:
                src.close()
                raise
            handles[path] = (src, vrt)
            while len(handles) > 24:
                old_src, old_vrt = handles.popitem(last=False)[1]
                old_vrt.close()
                old_src.close()
        handles.move_to_end(path)
        return handles[path][1].read(1, window=Window(x, y, width, height))

    def read(self, scene, y, x, height, width):
        scene = self.scenes.get(scene['asset_id'], scene)
        key = scene['key']
        names = LANDSAT_SR_BANDS if key == 'landsat' else BANDS[key]
        bands = np.stack([self.band(scene, b, y, x, height, width) for b in names])
        valid = np.isfinite(bands).all(axis=0)
        if key == 'landsat':
            qa = self.band(scene, 'QA_PIXEL', y, x, height, width)
            saturated = self.band(scene, 'QA_RADSAT', y, x, height, width)
            valid &= np.isfinite(qa) & np.isfinite(saturated)
            return landsat_l2_pixels(bands, np.nan_to_num(qa),
                                     np.nan_to_num(saturated), valid)
        if key == 's2':
            scl = self.band(scene, 'SCL', y, x, height, width)
            valid &= np.isfinite(scl) & ~np.isin(scl, [0, 1, 3, 8, 9, 10])
            valid &= (bands != 0).all(axis=0)
            params = scene['parameters']
            bands = np.stack([harmonize_s2(band, offset=params['offsets'][b],
                                         quantification=params['quantification'])
                              for b, band in zip(names, bands, strict=True)])
        return np.where(valid[None], bands, 0), valid


class LocalQualityCache(QualityCache):
    """Deduplicate block reads; do not repeatedly decode a source for every sample window."""

    def __init__(self, full, reader):
        super().__init__(full, tile_chunks=2)
        self.reader, self._inflight = reader, {}

    def get_tile(self, scene, ty, tx):
        key = (scene['asset_id'], ty, tx)
        with self.guard:
            try:
                return super().get_tile(scene, ty, tx)
            except FileNotFoundError:
                pass
            owner = key not in self._inflight
            if owner:
                self._inflight[key] = Future()
            future = self._inflight[key]
        if not owner:
            return future.result()
        try:
            if self.stopped.is_set():
                raise RuntimeError('Local preparation stopped after a worker failure')
            y, x, y1, x1 = self.tile_bounds(ty, tx)
            _, valid = self.reader.read(scene, y, x, y1 - y, x1 - x)
            valid = self.write_tile(scene, ty, tx, valid)
            future.set_result(valid)
            return valid
        except BaseException as exc:
            future.set_exception(exc)
            raise
        finally:
            with self.guard:
                self._inflight.pop(key, None)
