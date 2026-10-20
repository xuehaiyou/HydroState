"""Direct Earth Engine computePixels acquisition; no Drive dependency."""

from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from hydrostate.data.gee.backends.pixels import PixelSceneCache as SceneCache
from hydrostate.data.gee.catalog import catalog
from hydrostate.data.gee.contracts import locked
from hydrostate.data.gee.products import BANDS
from hydrostate.data.input_products import LEGACY_PROFILE
from hydrostate.data.regional.common import _prepare_windows
from hydrostate.inference.planning import windows


def prepare_date(root, plan, period, *, workers=4, min_free_gib=5):
    if plan["config"].get("input_profile", LEGACY_PROFILE) != LEGACY_PROFILE:
        raise ValueError("Regional pixels currently support the legacy L1 profile only")
    import ee

    root = Path(root)
    ee.Initialize(project=plan["config"]["gee_project"])
    ee.data.setDeadline(180000)
    dest = root / "manifests" / f"{period['time']}.parquet"
    if dest.exists():
        table = pd.read_parquet(dest)
        if len(table) != len(windows(plan)):
            raise ValueError("Incomplete date manifest")
        return table
    with locked(root / "prepare.lock", blocking=False):
        if (root / "recency_update_pending.json").exists():
            raise RuntimeError(
                "Resume the interrupted regional recency revision before preparation"
            )
        t = date.fromisoformat(period["time"])
        years = range((t - timedelta(days=63)).year, t.year + 1)
        scenes = {
            key: sorted(
                [s for year in years for s in catalog(root, plan, key, year, ee)],
                key=lambda s: (s["time"], s["asset_id"]),
                reverse=True,
            )
            for key in BANDS
        }
        cache = SceneCache(root, plan, ee=ee, min_free_gib=min_free_gib)
        return _prepare_windows(root, plan, period, scenes, cache, workers, min_free_gib)
