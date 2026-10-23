"""Finite cached-block cursor for rolling, fixed-capacity candidate pools."""

import logging
import time

from .catalog import CandidateCatalog, global_blocks

LOG = logging.getLogger(__name__)


class WaterCandidateSource:
    def __init__(self, ee, config, state=None):
        self.catalog = CandidateCatalog(ee, config)
        self.blocks = global_blocks(config)
        self.max_passes = config["max_dates_per_location"]
        state = state or {}
        self.index, self.pass_index = state.get("index", 0), state.get("pass_index", 0)

    def next_query(self, ledger, reserved):
        if self.index >= len(self.blocks):
            self.index = 0
            self.pass_index += 1
        if self.pass_index >= self.max_passes:
            return None
        index = self.index
        self.index += 1
        return index

    def fetch(self, index):
        started = time.monotonic()
        result = self.catalog.fetch((index, self.blocks[index]))
        LOG.info(
            "JRC candidates: block=%s count=%d seconds=%.2f",
            self.blocks[index],
            len(result),
            time.monotonic() - started,
        )
        return result

    def snapshot(self):
        return dict(index=self.index, pass_index=self.pass_index)
