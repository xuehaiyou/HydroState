"""Independent DW snow/ice proposals, bound to a real PML target interval."""

import logging
import math
import random
import time
from pathlib import Path

from hydrostate.data.gee.contracts import fingerprint
from hydrostate.data.gee.products import collection

from .catalog import cached_json, centre_sample, global_blocks
from .contract import product_end
from .sampling import ICE_GROUP, frequency_bin

LOG = logging.getLogger(__name__)


def candidate_expression(ee, area, start, end, config):
    """Cheap proposal only: at least one confident observation in this interval.

    Final eligibility is checked on the complete 30 m centre's mean coverage.
    JRC bands are metadata, deliberately unmasked so they cannot exclude ice.
    """
    images = collection(ee, "dw").filterBounds(area).filterDate(start, end).select("snow_and_ice")
    empty = ee.Image.constant(0).rename("snow_and_ice").updateMask(ee.Image.constant(0))
    ice = images.merge(ee.ImageCollection.fromImages([empty])).max()
    ice = ice.gte(config["ice_confidence"]).selfMask().rename("ice").toByte()
    observed = (
        ee.ImageCollection("JRC/GSW1_4/YearlyHistory")
        .select("waterClass")
        .max()
        .gt(0)
        .unmask(0, False)
        .rename("jrc_observed")
    )
    occurrence = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").unmask(0, False)
    climate = ee.Image(config["climate_asset"]).select(0).unmask(0, False).rename("climate_class")
    return ice.addBands([observed, occurrence, climate])


class IceCandidateSource:
    """One bounded query at a time, using current-run hints plus fresh regions.

    Hints only accelerate search; independent global searches bypass the JRC
    observation mask. Empty responses are cached as well as successful ones.
    """

    def __init__(self, ee, config, dates, rows=(), state=None):
        self.ee, self.config = ee, config
        self.rng = random.Random(config["seed"] + 1)
        self.dates = sorted(dates)
        self.blocks = global_blocks(config, seed_offset=1)
        self.block_index = 0
        self.hints, self.queried = {}, set()
        self.dispatched = 0
        self.stats = dict(queries=0, cache_hits=0, empty_queries=0, candidates=0, seconds=0.0)
        self.cache = Path(config["candidate_dir"]) / "dw_ice_v1"
        if state:
            self.block_index = state["block_index"]
            self.hints = state["hints"]
            self.queried = set(state["queried"])
            self.dispatched = state["dispatched"]
            self.stats = state["stats"]
            self.rng.setstate((state["rng"][0], tuple(state["rng"][1]), state["rng"][2]))
        for row in rows:
            self.observe(row)

    def snapshot(self):
        return dict(
            block_index=self.block_index,
            hints=dict(self.hints),
            queried=sorted(self.queried),
            dispatched=self.dispatched,
            stats=dict(self.stats),
            rng=self.rng.getstate(),
        )

    def observe(self, row):
        if row.get("max_snow_ice", 0) < self.config["min_center_ice_fraction"]:
            return
        x, y, step = row["block"]
        size = self.config["ice_search_degrees"]
        left = x + math.floor((row["lon"] - x) / size) * size
        bottom = y + math.floor((row["lat"] - y) / size) * size
        request = dict(
            block=[x, y, step],
            bounds=[left, bottom, min(x + step, left + size), min(y + step, bottom + size)],
            target_time=row["target_time"],
        )
        key = fingerprint(request)
        if key not in self.queried:
            self.hints[key] = request

    def next_query(self, ledger, reserved):
        # Three hint queries can be followed by a fresh-region search. Even a
        # fruitful JRC-derived hint never becomes the only allowed ice region.
        if self.hints and self.dispatched % 4 != 0:
            keys = list(self.hints)
            self.rng.shuffle(keys)
            key = min(keys, key=lambda k: self.priority(self.hints[k], ledger, reserved))
            query = self.hints.pop(key)
        else:
            if self.block_index >= len(self.blocks):
                if not self.hints:
                    return None
                key = next(iter(self.hints))
                query = self.hints.pop(key)
            else:
                x, y = self.blocks[self.block_index]
                self.block_index += 1
                step, size = (
                    self.config["candidate_block_degrees"],
                    self.config["ice_search_degrees"],
                )
                left = x + self.rng.randrange(math.ceil(step / size)) * size
                bottom = y + self.rng.randrange(math.ceil(step / size)) * size
                days = list(self.dates)
                self.rng.shuffle(days)
                day = min(
                    days,
                    key=lambda d: (
                        ledger.years[int(d[:4])]
                        + reserved.years[int(d[:4])]
                        + ledger.months[int(d[5:7])]
                        + reserved.months[int(d[5:7])]
                    ),
                )
                query = dict(
                    block=[x, y, step],
                    bounds=[left, bottom, min(x + step, left + size), min(y + step, bottom + size)],
                    target_time=day,
                )
                key = fingerprint(query)
        self.queried.add(key)
        self.dispatched += 1
        return query

    @staticmethod
    def priority(query, ledger, reserved):
        day, block = query["target_time"], tuple(query["block"])
        return (
            ledger.blocks[block] + reserved.blocks[block],
            ledger.years[int(day[:4])]
            + reserved.years[int(day[:4])]
            + ledger.months[int(day[5:7])]
            + reserved.months[int(day[5:7])],
        )

    def fetch(self, request):
        from pyproj import CRS

        started = time.monotonic()
        ee, config = self.ee, self.config
        identity = dict(
            request,
            source="GOOGLE/DYNAMICWORLD/V1",
            schema="dw_ice_v1",
            confidence=config["ice_confidence"],
            count=config["ice_candidates_per_query"],
            seed=config["candidate_seed"],
            climate=config["climate_asset"],
            margin=config["split_margin_m"],
        )
        key = fingerprint(identity)
        path = self.cache / f"{key}.json"
        hit = path.exists()

        def build():
            area = ee.Geometry.Rectangle(request["bounds"], geodesic=False)
            start = request["target_time"]
            image = candidate_expression(ee, area, start, product_end(start), config)
            return image.stratifiedSample(
                numPoints=0,
                classBand="ice",
                classValues=[1],
                classPoints=[config["ice_candidates_per_query"]],
                region=area,
                projection=CRS.from_epsg(6933).to_wkt(version="WKT1_GDAL"),
                scale=30,
                seed=int(key[:7], 16),
                geometries=True,
                tileScale=4,
            ).getInfo()["features"]

        result = []
        for feature in cached_json(path, build):
            if sample := centre_sample(feature, request["block"], config):
                observed = bool(sample["jrc_observed"])
                sample.update(
                    sampling_group=ICE_GROUP,
                    candidate_target_time=request["target_time"],
                    jrc_observed=observed,
                    frequency_bin=frequency_bin(sample["occurrence"]) if observed else -1,
                    occurrence=sample["occurrence"] if observed else -1,
                )
                result.append(sample)
        elapsed = time.monotonic() - started
        self.stats["queries"] += 1
        self.stats["cache_hits"] += int(hit)
        self.stats["empty_queries"] += int(not result)
        self.stats["candidates"] += len(result)
        self.stats["seconds"] += elapsed
        LOG.info(
            "DW ice candidates: date=%s bounds=%s count=%d seconds=%.2f cached=%s",
            request["target_time"],
            request["bounds"],
            len(result),
            elapsed,
            hit,
        )
        return result
