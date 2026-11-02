"""Area-consistent regional summaries and seasonal inference time-series figures."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd


def block_totals(water, land, et, sm, masks):
    """SM is a conditional land mean; missing surfaces never contribute zero SM."""
    w = masks['water'] & np.isfinite(water)
    e = masks['et_total'] & np.isfinite(et)
    s = (masks['soil_moisture'] & masks['land'] & np.isfinite(sm)
         & np.isfinite(land) & (land > 0))
    return np.array([
        water[w].sum(dtype='f8'), w.sum(), et[e].sum(dtype='f8'), e.sum(),
        (sm[s].astype('f8') * land[s]).sum(), land[s].sum(dtype='f8'), s.sum(),
    ], dtype='f8')


def summarize(product, output):
    import zarr
    from affine import Affine
    from rasterio.features import geometry_mask

    product, output = Path(product), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    group = zarr.open_group(str(product), mode='r')
    plan = json.loads((product.parent.parent / 'plan.json').read_text())
    if group.attrs['identity']['plan'] != plan['identity']:
        raise ValueError('Output product and input plan disagree')
    transform = Affine(*group.attrs['transform'])
    area_km2 = abs(transform.a * transform.e - transform.b * transform.d) / 1e6
    if plan['grid']['crs'] != 'EPSG:32650':
        raise ValueError('This area summary expects the Poyang projected metre grid')
    height, width = group['water'].shape[1:]
    inside = geometry_mask([plan['geometry']], (height, width), transform, invert=True)
    aoi_pixels = int(inside.sum())
    arrays = {t: group[t] for t in ('land', 'soil_moisture',
                                    'land_valid', 'soil_moisture_valid')}
    def summarize_date(index):
        start, end = group['time_bounds'][index].astype('datetime64[s]')
        day = str(start)[:10]
        # Export audits already contain exact full-resolution area sums for these
        # two metrics. Reuse them; only conditional SM needs a new joint reduction.
        audit = json.loads((product.parent / f'{product.stem}_{day}.json').read_text())
        if audit['target']['start'] != day or audit['audit']['inside_pixels'] != aoi_pixels:
            raise ValueError('Per-date export audit disagrees with the product grid')
        stats = audit['audit']['variables']
        water, et = stats['water'], stats['et_total']
        totals = np.array([water['mean'] * water['valid_pixels'], water['valid_pixels'],
                           et['mean'] * et['valid_pixels'], et['valid_pixels'],
                           0, 0, 0], dtype='f8')
        for y in range(0, height, 512):
            key = (int(index), slice(y, y + 512), slice(None))
            land, sm = arrays['land'][key], arrays['soil_moisture'][key]
            valid = (arrays['land_valid'][key].astype(bool)
                     & arrays['soil_moisture_valid'][key].astype(bool)
                     & inside[y:y + 512] & np.isfinite(land) & np.isfinite(sm) & (land > 0))
            totals[4] += (sm[valid].astype('f8') * land[valid]).sum()
            totals[5] += land[valid].sum(dtype='f8')
            totals[6] += valid.sum()
        if np.any(totals[[1, 3, 5]] == 0):
            raise ValueError(f'No valid support for completed date {index}')
        start, end = group['time_bounds'][index].astype('datetime64[s]')
        row = dict(
            start=str(start)[:10], end=str(end)[:10],
            midpoint=str(start + (end - start) // 2),
            water_area_km2=totals[0] * area_km2,
            et_total_mean_mm_day=totals[2] / totals[3],
            land_sm_mean_m3_m3=totals[4] / totals[5],
            aoi_area_km2=aoi_pixels * area_km2,
            water_valid_fraction=totals[1] / aoi_pixels,
            et_valid_fraction=totals[3] / aoi_pixels,
            sm_valid_area_km2=totals[6] * area_km2,
            sm_effective_land_area_km2=totals[5] * area_km2,
        )
        print(json.dumps(row), flush=True)
        return row

    # A single bounded pool overlaps filesystem latency; array handles are read-only.
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(summarize_date, np.flatnonzero(group['date_complete'][:])))
    if not rows:
        raise ValueError('No completed dates')
    pd.DataFrame(rows).sort_values('start').to_csv(output / 'timeseries.csv', index=False)
    (output / 'statistics.json').write_text(json.dumps(dict(
        source_product=str(product), product_identity=dict(group.attrs['identity']),
        samples=len(rows), nominal_pixel_area_km2=area_km2,
        water='sum(valid * water_fraction) * projected_pixel_area',
        et='area mean of valid total ET; no extra surface weighting of total ET',
        sm='sum(valid * land_fraction * land_SM) / sum(valid * land_fraction)',
        support='Per-date valid AOI; no display-only 5% mask; no fill for missing pixels',
        reduction='Reuse full-resolution export audits for water and ET; recompute weighted SM',
        interpretation='32 seasonal 8-day predictions, not a complete daily or annual series',
    ), indent=2))


def plot(csv, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    data = pd.read_csv(csv).sort_values('midpoint')
    dates = pd.to_datetime(data['midpoint'])
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(3, 1, figsize=(12, 8.5), sharex=True)
    fig.subplots_adjust(left=0.14, right=0.975, top=0.88, bottom=0.16, hspace=0.35)
    panels = [
        ('water_area_km2', 'Water area (km²)', '#2878A5', 'a  Water-covered area'),
        ('et_total_mean_mm_day', 'Mean total ET (mm/day)', '#C66A28', 'b  Regional total ET'),
        ('land_sm_mean_m3_m3', 'Mean land SM (m³/m³)', '#36836A', 'c  Land soil moisture'),
    ]
    for ax, (key, label, color, title) in zip(axes, panels):
        ax.plot(dates, data[key], color=color, marker='o', markersize=4, linewidth=1.6)
        ax.set_ylabel(label)
        ax.set_title(title, loc='left', fontsize=12, weight='bold', pad=8)
        ax.grid(axis='y', alpha=0.18)
        ax.margins(x=0.02, y=0.18)
    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    axes[-1].set_xlabel('Year (points at the midpoint of each 8-day period)')
    fig.suptitle('Poyang Lake | Seasonal retrieval time series, 2017–2024',
                 x=0.14, y=0.97, ha='left', fontsize=17, weight='bold')
    fig.text(0.14, 0.925, 'Shared-LoRA | Original study region | Four sampled periods per year',
             fontsize=11, color='#555555')
    fig.text(0.14, 0.07,
             'Water area: fraction-weighted sum. Total ET: area mean. '
             'SM: land-area-weighted mean.\n'
             'Lines connect seasonal samples; intervening dates are not inferred.',
             fontsize=10, color='#555555', linespacing=1.6)
    fig.canvas.draw()
    # All panels share identical plot edges and evenly spaced rows.
    positions = [ax.get_position() for ax in axes]
    assert np.ptp([p.x0 for p in positions]) < 1e-8
    assert np.ptp([p.width for p in positions]) < 1e-8
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)
