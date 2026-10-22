"""Tile projections and stable geographic train/validation/test splits."""

import hashlib
import json


def grid(lon, lat):
    from pyproj import Transformer

    zone = min(60, int((lon + 180) // 6) + 1)
    epsg = (32600 if lat >= 0 else 32700) + zone
    if lat >= 84:
        epsg = 3413
    elif lat < -80:
        epsg = 3031
    x, y = Transformer.from_crs(4326, epsg, always_xy=True).transform(lon, lat)
    left, top = round((x - 600) / 30) * 30, round((y + 600) / 30) * 30
    return f"EPSG:{epsg}", [10, 0, left, 0, -10, top]


def geographic_block(row):
    return f"{int((row['lon'] + 180) // 5)}:{int((row['lat'] + 90) // 5)}"


def geographic_split(row, seed):
    # Stable SHA256 assignment; independent of Python hash randomization.
    payload = json.dumps([seed, geographic_block(row)], sort_keys=True).encode()
    value = int(hashlib.sha256(payload).hexdigest()[:8], 16) % 100
    return "train" if value < 70 else "validation" if value < 85 else "test"
