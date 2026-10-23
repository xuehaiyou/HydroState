"""Hard centre quotas, soft temporal/climate balance, and spatial exclusion."""

import math
from collections import Counter, defaultdict
from datetime import date

ICE_GROUP = 7
SAMPLING_VERSION = "jrc7_dwice_v1"


def frequency_bin(value):
    if value is None or not math.isfinite(value) or not 0 <= value <= 100:
        return None
    return 0 if value == 0 else 6 if value == 100 else int(value // 20) + 1


def quotas(total):
    return {i: total // 8 + (i < total % 8) for i in range(8)}


def location_key(sample):
    return sample["crs"], sample["transform"][2], sample["transform"][5]


class SamplingLedger:
    """Only successfully validated shards consume quotas or location dates."""

    def __init__(self, rows=(), max_dates=4, min_days=40):
        self.max_dates, self.min_days = max_dates, min_days
        self.dates = defaultdict(list)
        self.counts, self.years, self.months, self.climates = (Counter() for _ in range(4))
        self.blocks, self.splits = Counter(), Counter()
        self.windows = defaultdict(list)
        for row in rows:
            self.add(row)

    def allows(self, sample, target):
        key, when = location_key(sample), date.fromisoformat(target)
        dates = self.dates.get(key, ())
        if len(dates) >= self.max_dates or any(
            abs((when - old).days) < self.min_days for old in dates
        ):
            return False
        # Check neighbouring spatial buckets; centres closer than 1.2 km would
        # create overlapping windows with near-duplicate temporal input support.
        crs, x, y = key
        bx, by = math.floor(x / 1200), math.floor(y / 1200)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for ox, oy, day in self.windows.get((crs, bx + dx, by + dy), ()):
                    if (
                        abs(x - ox) < 1200
                        and abs(y - oy) < 1200
                        and abs((when - day).days) < self.min_days
                    ):
                        return False
        return True

    def add(self, sample):
        when = date.fromisoformat(sample["target_time"])
        key = location_key(sample)
        self.dates[key].append(when)
        crs, x, y = key
        self.windows[(crs, math.floor(x / 1200), math.floor(y / 1200))].append((x, y, when))
        self.counts[sample["sampling_group"]] += 1
        self.years[when.year] += 1
        self.months[when.month] += 1
        self.climates[sample.get("climate_class", 0)] += 1
        self.blocks[tuple(sample.get("block", ()))] += 1
        self.splits[sample.get("split", "")] += 1


def inside_split_block(lon, lat, margin_m=35000):
    """Exclude full approximate 33 km footprints on either side of split edges.

    Conservative 35 km margins also separate adjacent buffered supports, instead
    of merely putting centres in different blocks. Climate is not an eligibility mask.
    """
    dx = min((lon + 180) % 5, 5 - (lon + 180) % 5) * 111320 * math.cos(math.radians(lat))
    dy = min((lat + 90) % 5, 5 - (lat + 90) % 5) * 111320
    return min(dx, dy) >= margin_m
