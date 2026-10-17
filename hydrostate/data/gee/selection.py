"""Observation selection retains the existing temporal and quality contract."""

import json
from datetime import UTC, datetime

from shapely.geometry import box

from hydrostate.inference.planning import overlap_records, window_geometry

from .products import BANDS, slots


def epoch(day):
    return int(datetime.fromisoformat(str(day)).replace(tzinfo=UTC).timestamp())


def select_window(plan, period, window, scenes, cache):
    cfg, selected, orbit = plan["config"], {}, None
    footprint = window_geometry(plan["grid"], window)
    for slot, (start, end) in reversed(list(enumerate(slots(period["time"])))):
        a, b = epoch(start), epoch(end)
        for key in BANDS:
            choice = None
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
                # Evaluate the 30 m pixel centres on our explicitly aligned grid.
                fraction = float(valid[1::3, 1::3].mean())
                if fraction >= cfg["min_valid_fraction"]:
                    choice = dict(scene, valid_fraction=fraction)
                    break
            selected[f"{key}_{slot}"] = choice
            if key == "s1" and choice:
                orbit = choice["orbit"]
    active = {name.rsplit("_", 1)[1] for name, item in selected.items() if item}
    latest = max((s["time"] for s in selected.values() if s), default=0)
    # Match the training requirement at day resolution (acquisitions on t are allowed).
    age = max(0, (epoch(period["time"]) - latest) / 86400) if latest else None
    accepted = (
        len(active) >= cfg["min_active_slots"] and age is not None and age <= cfg["latest_days"]
    )
    return dict(
        window,
        time=period["time"],
        accepted=accepted,
        active_slots=len(active),
        latest_age_days=age,
        selections=json.dumps(selected, sort_keys=True),
        **{
            k: json.dumps(v)
            for k, v in overlap_records(plan["grid"], window, period, plan["training"]).items()
        },
    )
