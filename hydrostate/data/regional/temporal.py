"""Explicit regional observation windows, independent of training sample defaults."""

from datetime import date, timedelta

PAST = "past_16d_v1"
CENTERED = "centered_8d_v1"
OFFSETS = {
    PAST: ((-63, -47), (-47, -31), (-31, -15), (-15, 1)),
    CENTERED: ((-16, -8), (-8, 0), (0, 8), (8, 16)),
}


def policy(config):
    name = config.get("temporal_sampling", PAST)
    if name not in OFFSETS:
        raise ValueError(f"Unknown regional temporal_sampling: {name}")
    if name == CENTERED and config.get("latest_days") is not None:
        raise ValueError("centered_8d_v1 uses slot support; set latest_days=null")
    return name


def regional_slots(target, config):
    """Half-open UTC calendar intervals; centered mode contains exactly 32 days."""
    t = date.fromisoformat(target)
    return [
        ((t + timedelta(days=a)).isoformat(), (t + timedelta(days=b)).isoformat())
        for a, b in OFFSETS[policy(config)]
    ]


def context_offsets(config):
    intervals = OFFSETS[policy(config)]
    return intervals[0][0], intervals[-1][1]
