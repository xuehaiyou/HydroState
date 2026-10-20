"""Third regional acquisition backend: local files only, no GEE/Drive initialization."""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from hydrostate.data.gee.contracts import atomic_json, fingerprint, input_shape, locked
from hydrostate.data.gee.products import BANDS
from hydrostate.data.gee.region_plan import selected_chunks
from hydrostate.data.gee.selection import epoch
from hydrostate.data.input_products import L2_PROFILE
from hydrostate.data.local.catalog import catalog
from hydrostate.data.local.reader import LocalQualityCache, RasterReader
from hydrostate.data.regional.common import _prepare_windows, _publish_table
from hydrostate.data.regional.temporal import regional_slots
from hydrostate.inference.planning import windows
from hydrostate.storage.scene_cache import SceneCache

LOG = logging.getLogger(__name__)


def prepare_region_local(root, plan, periods, settings, *, workers=4, min_free_gib=20):
    if plan["config"].get("input_profile") != L2_PROFILE:
        raise ValueError("Local L2 inputs require a separate plan with input_profile=" + L2_PROFILE)
    root = Path(root)
    with locked(root / "prepare.lock", blocking=False):
        indexed = catalog(root, plan, settings)
        full = SceneCache(root, plan, min_free_gib=min_free_gib)
        contract = dict(
            plan=fingerprint(plan),
            catalog=fingerprint(indexed),
            preprocessing=full.contract,
            backend="local",
            version=1,
        )
        contract_path = root / "local_contract.json"
        if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
            raise ValueError("Local preparation contract changed; use a new root")
        atomic_json(contract_path, contract)
        reader = RasterReader(plan["grid"], indexed["scenes"])
        quality = LocalQualityCache(full, reader)
        height, width = input_shape(plan["grid"])
        expected_windows = len(windows(plan))

        def prepare_chunk(item):
            scene, cy, cx = item
            y, x = cy * full.chunk, cx * full.chunk
            values, valid = reader.read(
                scene, y, x, min(full.chunk, height - y), min(full.chunk, width - x)
            )
            full.write_chunk(scene, cy, cx, values, valid)

        for period in periods:
            dest = root / "manifests" / f"{period['time']}.parquet"
            if dest.exists():
                if len(pd.read_parquet(dest)) != expected_windows:
                    raise ValueError("Incomplete local date manifest")
                continue
            started = time.monotonic()
            intervals = regional_slots(period["time"], plan["config"])
            first, last = intervals[0][0], intervals[-1][1]
            scenes = {
                key: [
                    {k: s[k] for k in ("key", "asset_id", "time", "orbit", "bounds")}
                    for s in indexed["scenes"]
                    if s["key"] == key and epoch(first) <= s["time"] < epoch(last)
                ]
                for key in BANDS
            }
            quality.refresh_full()
            table = _prepare_windows(
                root, plan, period, scenes, quality, workers, min_free_gib, publish=False
            )
            demands = selected_chunks(full, table.to_dict("records"))
            pending = []
            for demand in demands:
                done = full.done(demand["scene"])
                pending.extend(
                    (demand["scene"], cy, cx)
                    for cy, cx in sorted(demand["chunks"])
                    if not done[cy, cx]
                )
            # map consumes descriptors only; each worker holds one bounded spectral block.
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for count, _ in enumerate(pool.map(prepare_chunk, pending), 1):
                    if count % 100 == 0:
                        LOG.info("%s cached %d/%d blocks", period["time"], count, len(pending))
            quality.verify_full(demands)
            report = dict(
                time=period["time"],
                backend="local",
                input_profile=L2_PROFILE,
                windows=len(table),
                accepted=int(table.accepted.sum()),
                candidates={key: len(value) for key, value in scenes.items()},
                missing_sensors=[key for key, value in scenes.items() if not value],
                new_chunks=len(pending),
                elapsed_seconds=time.monotonic() - started,
                source_equivalence="S1/S2 cross-source validation pending",
            )
            report_path = root / "reports" / f"local_{period['time']}.json"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_json(report_path, report)
            _publish_table(table, dest)
            LOG.info("Prepared local inputs: %s", report)
