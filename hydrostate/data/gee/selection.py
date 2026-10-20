"""Observation selection retains the existing temporal and quality contract."""

import json
from datetime import UTC, datetime

from shapely.geometry import box

from hydrostate.data.regional.temporal import CENTERED, context_offsets, policy, regional_slots
from hydrostate.inference.planning import overlap_records, window_geometry

from .products import BANDS


def epoch(day):
    return int(datetime.fromisoformat(str(day)).replace(tzinfo=UTC).timestamp())


def select_window(plan, period, window, scenes, cache):
    cfg, selected, orbit = plan["config"], {}, None
    footprint = window_geometry(plan["grid"], window)
    intervals = regional_slots(period["time"], cfg)
    target = epoch(period["time"])
    centered = policy(cfg) == CENTERED
    if centered:
        scenes = {
            key: sorted(group, key=lambda s: (abs(s["time"] - target), -s["time"], s["asset_id"]))
            for key, group in scenes.items()
        }

    def choose(key, a, b, orbit):
        for scene in scenes[key]:
            if not a <= scene["time"] < b:
                continue
            if key == "s1" and orbit is not None and scene["orbit"] != orbit:
                continue
            if not box(*scene["bounds"]).intersects(footprint):
                continue
            valid = (
                cache.read_valid(scene, window)
                if hasattr(cache, "read_valid")
                else cache.read(scene, window)[1]
            )
            fraction = float(valid[1::3, 1::3].mean())
            if fraction >= cfg["min_valid_fraction"]:
                return dict(scene, valid_fraction=fraction)
        return None

    if centered:
        # Anchor the shared S1 orbit to the nearest qualified observation, not to
        # the farthest future slot just because the legacy loop runs backwards.
        anchor = choose("s1", epoch(intervals[0][0]), epoch(intervals[-1][1]), None)
        if anchor:
            orbit = anchor["orbit"]
    for slot, (start, end) in reversed(list(enumerate(intervals))):
        a, b = epoch(start), epoch(end)
        for key in BANDS:
            choice = choose(key, a, b, orbit)
            selected[f"{key}_{slot}"] = choice
            if key == "s1" and choice:
                orbit = choice["orbit"]
    active = {name.rsplit("_", 1)[1] for name, item in selected.items() if item}
    latest = max((s["time"] for s in selected.values() if s), default=0)
    # Match the training requirement at day resolution (acquisitions on t are allowed).
    age = (target - latest) / 86400 if latest else None
    if age is not None and not centered:
        age = max(0, age)
    limit = cfg["latest_days"]
    accepted = (
        len(active) >= cfg["min_active_slots"] and age is not None
        and (limit is None or age <= limit)
    )
    diagnostics = {}
    if centered:
        offsets = {k: (s["time"] - target) / 86400 for k, s in selected.items() if s}
        diagnostics = dict(
            nearest_observation_distance_days=min(map(abs, offsets.values())) if offsets else None,
            has_before_observation=any(v < 0 for v in offsets.values()),
            has_after_observation=any(v > 0 for v in offsets.values()),
            observation_offsets_days=json.dumps(offsets, sort_keys=True),
        )
    return dict(
        window,
        time=period["time"],
        accepted=accepted,
        active_slots=len(active),
        latest_age_days=age,
        selections=json.dumps(selected, sort_keys=True),
        **diagnostics,
        **{
            k: json.dumps(v)
            for k, v in overlap_records(
                plan["grid"], window, period, plan["training"],
                input_offsets=context_offsets(cfg),
            ).items()
        },
    )
