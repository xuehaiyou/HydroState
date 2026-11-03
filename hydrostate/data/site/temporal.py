"""UTC support of morning point measurements and interval-mean latent heat flux."""

from calendar import isleap
from datetime import UTC, date, datetime, timedelta

import numpy as np

DAY = 86400
LATENT_HEAT = 2450000.0


def epoch(value):
    stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return (
        int(stamp.replace(tzinfo=UTC).timestamp())
        if stamp.tzinfo is None
        else int(stamp.timestamp())
    )


def windows(start, end, *, as_of):
    """PML-style calendar independent of PML availability; require ended right context."""
    start, end, ceiling = date.fromisoformat(start), date.fromisoformat(end), epoch(as_of)
    for year in range(start.year, end.year + 1):
        cursor, next_year = date(year, 1, 1), date(year + 1, 1, 1)
        while cursor < next_year:
            stop = min(cursor + timedelta(days=8), next_year)
            if cursor >= start and stop <= end and epoch(cursor + timedelta(days=24)) <= ceiling:
                yield cursor.isoformat(), stop.isoformat()
            cursor += timedelta(days=8)


def morning_sm(times, values, good, longitude, start, end, *, coverage=0.8):
    """One nearest observation per mean-solar 06:00 day (ties prefer earlier UTC)."""
    times, values = np.asarray(times, dtype="int64"), np.asarray(values, dtype=float)
    start, end = epoch(start), epoch(end)
    offset = float(longitude) * 240  # Mean solar time, deliberately not civil/DST time.
    first = int(np.floor((start + offset) / DAY)) - 1
    last = int(np.floor((end + offset) / DAY)) + 1
    days = np.arange(first, last + 1)
    expected = days * DAY + 6 * 3600 - offset
    days, expected = (
        days[(expected >= start) & (expected < end)],
        expected[(expected >= start) & (expected < end)],
    )
    valid = (
        np.asarray(good, bool)
        & np.isfinite(values)
        & (values >= 0)
        & (values <= 1)
        & (times >= start)
        & (times < end)
    )
    rows = []
    for day, ideal in zip(days, expected, strict=True):
        candidates = np.flatnonzero(valid & (np.abs(times - ideal) <= 3600))
        if not len(candidates):
            continue
        pick = candidates[np.lexsort((times[candidates], np.abs(times[candidates] - ideal)))[0]]
        rows.append(
            dict(
                solar_date=datetime.fromtimestamp(int(day * DAY), UTC).date().isoformat(),
                utc=int(times[pick]),
                value=float(values[pick]),
                offset_seconds=float(times[pick] - ideal),
            )
        )
    count = len(rows)
    return dict(
        value=float(np.mean([r["value"] for r in rows])) if rows else None,
        valid=bool(len(days) and count >= coverage * len(days)),
        valid_days=count,
        expected_days=len(days),
        morning=rows,
        time_definition="mean_solar_06_nearest_within_1h",
    )


def daylight_intervals(start, end, latitude, longitude):
    """NOAA sunrise approximation (90.833 degrees); polar day/night handled explicitly.

    https://gml.noaa.gov/grad/solcalc/solareqns.PDF
    Declination/equation of time evaluated daily; no claim of subminute astronomical precision.
    """
    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        raise ValueError("Invalid site coordinates")
    cursor = datetime.fromtimestamp(start, UTC).date() - timedelta(days=2)
    stop = datetime.fromtimestamp(end, UTC).date() + timedelta(days=2)
    result = []
    lat = np.deg2rad(latitude)
    while cursor <= stop:
        gamma = 2 * np.pi / (366 if isleap(cursor.year) else 365) * (cursor.timetuple().tm_yday - 1)
        eq = 229.18 * (
            0.000075
            + 0.001868 * np.cos(gamma)
            - 0.032077 * np.sin(gamma)
            - 0.014615 * np.cos(2 * gamma)
            - 0.040849 * np.sin(2 * gamma)
        )
        dec = (
            0.006918
            - 0.399912 * np.cos(gamma)
            + 0.070257 * np.sin(gamma)
            - 0.006758 * np.cos(2 * gamma)
            + 0.000907 * np.sin(2 * gamma)
            - 0.002697 * np.cos(3 * gamma)
            + 0.00148 * np.sin(3 * gamma)
        )
        hour_cos = (np.cos(np.deg2rad(90.833)) - np.sin(lat) * np.sin(dec)) / max(
            1e-15, np.cos(lat) * np.cos(dec)
        )
        if hour_cos < 1:
            if hour_cos <= -1:
                left, right = max(start, epoch(cursor)), min(end, epoch(cursor) + DAY)
            else:
                half = np.rad2deg(np.arccos(hour_cos)) * 240
                noon = epoch(cursor) + (720 - 4 * longitude - eq) * 60
                left, right = max(start, noon - half), min(end, noon + half)
            if right > left:
                # Adjacent polar-day estimates can overlap: union, never double count.
                if result and left <= result[-1][1]:
                    result[-1] = (result[-1][0], max(right, result[-1][1]))
                else:
                    result.append((left, right))
        cursor += timedelta(days=1)
    return result


def interval_et(
    starts,
    ends,
    le,
    quality,
    latitude,
    longitude,
    start,
    end,
    *,
    min_valid=0.8,
    min_measured=0.5,
    min_day_night=0.8,
    support_valid=None,
):
    """FLUXNET HH/HR QC 0/1 only; all coverage denominators include missing time."""
    starts, ends = np.asarray(starts, float), np.asarray(ends, float)
    le, quality = np.asarray(le, float), np.asarray(quality, float)
    if not (starts.shape == ends.shape == le.shape == quality.shape):
        raise ValueError("Interval arrays must have identical shapes")
    if (
        np.any(~np.isfinite(starts + ends))
        or np.any(ends <= starts)
        or np.any(ends - starts > 3600)
        or np.any(starts[1:] < ends[:-1])
    ):
        raise ValueError("Expected ordered nonoverlapping HH/HR intervals")
    start, end = epoch(start), epoch(end)
    left, right = np.maximum(starts, start), np.minimum(ends, end)
    duration = np.maximum(0, right - left)
    valid = np.isfinite(le) & np.isin(quality, [0, 1]) & (duration > 0)
    if support_valid is not None:
        valid &= np.asarray(support_valid, bool)
    daylight = daylight_intervals(start, end, latitude, longitude)
    day_seconds = np.zeros(len(starts))
    for a, b in daylight:
        day_seconds += np.maximum(0, np.minimum(right, b) - np.maximum(left, a))
    full = end - start
    full_day = min(full, sum(b - a for a, b in daylight))
    full_night = full - full_day
    effective = float(duration[valid].sum())
    measured = float(duration[valid & (quality == 0)].sum())
    day_valid = float(day_seconds[valid].sum())
    night_valid = float((duration - day_seconds)[valid].sum())
    day_fraction = day_valid / full_day if full_day > 1 else None
    night_fraction = night_valid / full_night if full_night > 1 else None
    value = (
        float(np.sum(le[valid] * duration[valid]) / effective * DAY / LATENT_HEAT)
        if effective
        else None
    )
    reasons = []
    if effective < min_valid * full:
        reasons.append("effective_duration")
    if measured < min_measured * full:
        reasons.append("measured_duration")
    if day_fraction is not None and day_fraction < min_day_night:
        reasons.append("day_coverage")
    if night_fraction is not None and night_fraction < min_day_night:
        reasons.append("night_coverage")
    if value is None or value < 0:
        reasons.append("missing_or_negative_window_et")
    return dict(
        value=value,
        valid=not reasons,
        reasons=reasons,
        valid_fraction=effective / full,
        measured_fraction=measured / full,
        filled_fraction=(effective - measured) / full,
        day_fraction=day_fraction,
        night_fraction=night_fraction,
        effective_seconds=effective,
        window_seconds=full,
        daylight_method="NOAA_daily_declination_90.833deg",
    )
