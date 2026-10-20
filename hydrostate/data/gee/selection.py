"""Observation selection retains the existing temporal and quality contract."""

import json
from datetime import UTC, datetime

from shapely.geometry import box

from hydrostate.data.temporal_sampling import (
    CENTERED,
    accept_observations,
    context_offsets,
    policy,
    scene_order,
    select_observations,
)
from hydrostate.inference.planning import overlap_records, window_geometry

from .products import BANDS


def epoch(day):
    return int(datetime.fromisoformat(str(day)).replace(tzinfo=UTC).timestamp())


def select_window(plan, period, window, scenes, cache):
    cfg = plan["config"]
    footprint = window_geometry(plan["grid"], window)
    target = epoch(period["time"])
    centered = policy(cfg) == CENTERED
    if centered:
        scenes = {
            key: sorted(group, key=lambda s: scene_order(s["time"], s["asset_id"], target))
            for key, group in scenes.items()
        }

    def choose(key, a, b, orbit):
        a, b = epoch(a), epoch(b)
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

    selected = select_observations(
        period["time"], cfg, BANDS, choose, lambda s: s.get("orbit"),
    )
    active = {name.rsplit("_", 1)[1] for name, item in selected.items() if item}
    latest = max((s["time"] for s in selected.values() if s), default=0)
    # Match the training requirement at day resolution (acquisitions on t are allowed).
    age = (target - latest) / 86400 if latest else None
    if age is not None and not centered:
        age = max(0, age)
    accepted = accept_observations(
        {k: s["time"] for k, s in selected.items() if s}, target, cfg,
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
