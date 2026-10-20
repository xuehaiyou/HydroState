"""Reuse a frozen AOI without requesting lake geometry over the network."""

import copy
import json
from pathlib import Path

from hydrostate.data.gee.contracts import atomic_json, fingerprint, locked
from hydrostate.data.input_products import LEGACY_PROFILE, profile_bands
from hydrostate.data.regional.temporal import policy
from hydrostate.inference.planning import periods, windows


def initialize_from_template(root, config_path, template_root, *, reuse_input_cache=False):
    root, template_root = Path(root), Path(template_root)
    if root.resolve() == template_root.resolve():
        raise ValueError("A revised regional plan requires a new output root")
    source = json.loads((template_root / "plan.json").read_text())
    config = json.loads(Path(config_path).read_text())
    profile_bands(config.get("input_profile", LEGACY_PROFILE))
    policy(config)
    for key in (
        "crs",
        "input_size",
        "output_resolution_m",
        "core_size",
        "stride_size",
        "lake_collection",
        "lake_id",
        "buffer_m",
        "training_manifest",
    ):
        if config.get(key) != source["config"].get(key):
            raise ValueError(f"Template {key} differs; initialize a new AOI instead")
    result = copy.deepcopy(source)
    result["config"] = config
    result["periods"] = periods(config["start"], config["end"])
    if source["identity"].get("pilot"):
        dates = set(config["pilot_dates"])
        result["periods"] = [p for p in result["periods"] if p["time"] in dates]
    result["identity"] = dict(
        config=config,
        template_plan_sha256=fingerprint(source),
        pilot=source["identity"].get("pilot", False),
    )
    result.pop("plan_hash", None)
    result["plan_hash"] = fingerprint(result)
    root.mkdir(parents=True, exist_ok=True)
    with locked(root / "prepare.lock", blocking=False):
        path = root / "plan.json"
        if path.exists() and json.loads(path.read_text()) != result:
            raise ValueError("Existing plan differs; use a new root")
        if reuse_input_cache:
            from hydrostate.storage.scene_cache import SceneCache

            previous = SceneCache(template_root, source)
            current = SceneCache(root, result)
            if previous.contract != current.contract or previous.chunk != current.chunk:
                raise ValueError("Shared caches require identical input profile/grid/normalization")
            if not (template_root / "input_cache").is_dir():
                raise FileNotFoundError("Template has no reusable input cache")
            links = {}
            for name in ("input_cache", "quality_cache", "catalogs", "catalog"):
                origin, destination = template_root / name, root / name
                if not origin.exists():
                    continue
                if destination.exists() or destination.is_symlink():
                    if not destination.is_symlink() or destination.resolve() != origin.resolve():
                        raise ValueError(f"Cache reuse destination already exists: {destination}")
                else:
                    destination.symlink_to(origin.resolve(), target_is_directory=True)
                links[name] = str(origin.resolve())
            atomic_json(
                root / "cache_reuse.json",
                dict(
                    preprocessing=current.contract,
                    template_root=str(template_root.resolve()),
                    shared_directories=links,
                    note="Shared pixel/QA/asset caches; independent selections and job receipts",
                ),
            )
        atomic_json(path, result)
        import pandas as pd

        pd.DataFrame(windows(result)).to_parquet(root / "windows.parquet", index=False)
        if (template_root / "aoi.geojson").exists():
            (root / "aoi.geojson").write_bytes((template_root / "aoi.geojson").read_bytes())
    return result
