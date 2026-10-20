"""Shared observation intervals for training samples and regional inputs."""

import json
from datetime import date, timedelta

PAST = "past_16d_v1"
CENTERED = "centered_8d_v1"
OFFSETS = {
    PAST: ((-63, -47), (-47, -31), (-31, -15), (-15, 1)),
    CENTERED: ((-16, -8), (-8, 0), (0, 8), (8, 16)),
}
LEGACY_SAMPLE_CONFIG = dict(
    temporal_sampling=PAST, min_active_slots=2, min_valid=0.5, latest_days=3,
)


def policy(config):
    name = config.get("temporal_sampling", PAST)
    if name not in OFFSETS:
        raise ValueError(f"Unknown temporal_sampling: {name}")
    if name == CENTERED and config.get("latest_days") is not None:
        raise ValueError("centered_8d_v1 uses slot support; set latest_days=null")
    return name


def observation_slots(target, config):
    """Half-open UTC calendar intervals; centered mode contains exactly 32 days."""
    t = date.fromisoformat(target)
    return [
        ((t + timedelta(days=a)).isoformat(), (t + timedelta(days=b)).isoformat())
        for a, b in OFFSETS[policy(config)]
    ]


def context_offsets(config):
    intervals = OFFSETS[policy(config)]
    return intervals[0][0], intervals[-1][1]


def scene_order(timestamp, asset_id, target):
    """Nearest first, then newer on a tie, then a reproducible asset identity."""
    return abs(timestamp - target), -timestamp, asset_id


def select_observations(target, config, modalities, choose, orbit_of):
    """Shared slot/orbit selection; choose returns one qualified scene or None."""
    intervals = observation_slots(target, config)
    centered = policy(config) == CENTERED
    selected, orbit = {}, None
    if centered:
        anchor = choose("s1", intervals[0][0], intervals[-1][1], None)
        if anchor:
            orbit = orbit_of(anchor)
            if orbit is None:
                raise ValueError("Selected S1 scene lacks its relative orbit")
    for slot, (start, end) in reversed(list(enumerate(intervals))):
        for key in modalities:
            choice = (
                None if centered and key == "s1" and orbit is None
                else choose(key, start, end, orbit if key == "s1" else None)
            )
            selected[f"{key}_{slot}"] = choice
            if key == "s1" and choice:
                orbit = orbit_of(choice)
    return selected


def accept_observations(times, target, config):
    """Times are seconds keyed by sensor_slot; two sensors in one slot count once."""
    active = {key.rsplit("_", 1)[1] for key in times}
    if not times or len(active) < config["min_active_slots"]:
        return False
    limit = config["latest_days"]
    return limit is None or (target - max(times.values())) / 86400 <= limit


def sample_temporal_config(config):
    """Resolve legacy metadata explicitly; never reinterpret old samples."""
    result = {key: config.get(key, value) for key, value in LEGACY_SAMPLE_CONFIG.items()}
    policy(result)
    count = result["min_active_slots"]
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 4:
        raise ValueError("min_active_slots must be an integer in [1,4]")
    if not 0 < result["min_valid"] <= 1:
        raise ValueError("min_valid must be in (0,1]")
    limit = result["latest_days"]
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 0):
        raise ValueError("latest_days must be a nonnegative integer or null")
    return result


def sample_row_config(row):
    """Read the persisted sample contract, defaulting only unversioned old rows."""
    config = row.get("input_temporal_config")
    if isinstance(config, str):
        config = json.loads(config)
    if not isinstance(config, dict):
        name = row.get("temporal_sampling")
        if isinstance(name, str) and name != PAST:
            raise ValueError("Versioned sample lacks input_temporal_config")
        config = LEGACY_SAMPLE_CONFIG
    result = sample_temporal_config(config)
    name = row.get("temporal_sampling")
    if isinstance(name, str) and name != result["temporal_sampling"]:
        raise ValueError("Sample temporal_sampling disagrees with input_temporal_config")
    return result
