#!/usr/bin/env python3
"""Collect the current five-slot product-supervised dataset (old formats rejected)."""

import argparse
import json
import logging
import runpy
from pathlib import Path

from hydrostate.data.gee.contracts import fingerprint
from hydrostate.storage.packs import file_hash

from .sampling import SAMPLING_VERSION

DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / "configs/data/collection/retrieval.py"


def resolve_config(args):
    config = {k: v for k, v in runpy.run_path(str(args.config)).items() if not k.startswith("_")}
    for key, value in (
        ("output_dir", args.output),
        ("samples", args.samples),
        ("project", args.project),
        ("candidate_dir", getattr(args, "candidate_dir", None)),
        ("concurrency", getattr(args, "concurrency", None)),
    ):
        if value is not None:
            config[key] = str(value) if isinstance(value, Path) else value
    required = {
        "samples",
        "start_year",
        "end_year",
        "max_dates_per_location",
        "min_date_gap_days",
        "concurrency",
        "candidate_block_batch",
        "candidate_pool_factor",
        "ice_candidates_per_query",
        "ice_candidate_buffer",
        "ice_search_degrees",
        "min_center_ice_fraction",
        "split_margin_m",
        "water_confidence",
        "ice_confidence",
        "land_confidence",
        "land_max_water_ice_probability",
        "output_dir",
        "candidate_dir",
        "project",
        "seed",
        "split_seed",
        "candidate_seed",
        "candidates_per_bin",
        "candidate_block_degrees",
        "climate_asset",
        "min_active_slots",
        "min_input_fraction",
        "min_known_fraction",
        "max_water_ice_fraction",
        "min_tile_land_fraction",
        "min_cover_pixels",
    }
    if missing := required - config.keys():
        raise ValueError(f"Not a current retrieval collection config; missing: {sorted(missing)}")
    if config["samples"] < 8 or config["start_year"] > config["end_year"]:
        raise ValueError("Require at least eight samples and an ordered year range")
    if not 1 <= config["max_dates_per_location"] <= 4 or config["min_date_gap_days"] < 40:
        raise ValueError("At most four dates per location, separated by at least 40 days")
    if config["concurrency"] < 1 or config["candidate_block_batch"] < 1:
        raise ValueError("Concurrency and block batch must be positive")
    if not isinstance(config["candidate_pool_factor"], int) or config["candidate_pool_factor"] < 1:
        raise ValueError("Candidate pool factor must be a positive integer")
    if min(config["ice_candidates_per_query"], config["ice_candidate_buffer"]) < 1:
        raise ValueError("Ice candidate batch and buffer must be positive")
    if not 0 < config["min_center_ice_fraction"] <= 1:
        raise ValueError("Centre ice fraction must be in (0,1]")
    if not 0 < config["ice_search_degrees"] <= config["candidate_block_degrees"]:
        raise ValueError("Ice search boxes must fit inside candidate blocks")
    if config["split_margin_m"] < 35000:
        raise ValueError("Split margins must separate approximate SMAP supports")
    if any(
        not 0 < config[key] < 1
        for key in (
            "water_confidence",
            "ice_confidence",
            "land_confidence",
            "land_max_water_ice_probability",
        )
    ):
        raise ValueError("Class confidence thresholds must be in (0,1)")
    if config["ice_confidence"] <= 0.5 or config["land_max_water_ice_probability"] >= min(
        config["water_confidence"], config["ice_confidence"]
    ):
        raise ValueError("Cover thresholds must define mutually exclusive surface classes")
    repo = Path(__file__).resolve().parents[3]
    sources = [
        *sorted((repo / "hydrostate/data/retrieval").glob("*.py")),
        repo / "hydrostate/data/native_support.py",
        repo / "hydrostate/data/gee/products.py",
        repo / "hydrostate/data/spatial.py",
        repo / "hydrostate/storage/writer.py",
        repo / "hydrostate/storage/schema.py",
    ]
    config["sampling_version"] = SAMPLING_VERSION
    config["producer_hash"] = fingerprint({str(p.relative_to(repo)): file_hash(p) for p in sources})
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--project")
    parser.add_argument("--candidate-dir", type=Path)
    parser.add_argument(
        "--concurrency", type=int, help="Maximum concurrent queries and samples (default: config)"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = resolve_config(args)
    if args.dry_run:
        print(json.dumps(config, indent=2))
        return
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from .collector import collect

    collect(config)


if __name__ == "__main__":
    main()
