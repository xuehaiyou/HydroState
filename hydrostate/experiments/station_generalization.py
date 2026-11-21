"""Separate between-station and within-station skill on matched observation windows."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from hydrostate.experiments.cgc_gate_report import metrics


def attach_station_identity(predictions, release, split):
    from hydrostate.data.site.product_baseline import observations

    keys = ['task', 'sample_id', 'observation']
    metadata = []
    for item in observations(release, split=split):
        record = item['record']
        metadata.append(dict(task='site_' + item['task'], sample_id=item['sample_id'],
                             observation=item['observation_index'], site=item['spatial_group_id'],
                             station=record['group_id'], date=record['target_time'],
                             label=item['original_target']))
    data = predictions.merge(pd.DataFrame(metadata), on=[*keys, 'site'], how='left',
                             validate='many_to_one')
    if data[['station', 'date']].isna().any().any() or not np.allclose(data.target, data.label):
        raise ValueError('Predictions and native station labels disagree')
    return data


def station_report(data, draws=2000, seed=3407):
    """Return window-weighted within R, station-equal between R and macro RMSE."""
    keys = ['site', 'station', 'date', 'sample_id', 'observation']
    summaries, stations = [], []
    for task, records in data.groupby('task'):
        pred = records.pivot(index=keys, columns='mode', values='pred')
        labels = records.groupby(keys).target
        if (labels.nunique() != 1).any() or not np.isfinite(pred).all().all():
            raise ValueError('Modes must have identical paired observations')
        pred['target'] = labels.first()
        if not np.isfinite(pred.target).all():
            raise ValueError('Nonfinite station target')
        windows = pred.groupby(level=['site','station','date']).mean().reset_index()
        if windows.groupby('station').site.nunique().max() != 1:
            raise ValueError('A station spans multiple bootstrap clusters')
        sites = sorted(windows.site.unique())
        rng = np.random.default_rng(seed)
        weights = rng.multinomial(len(sites), np.full(len(sites), 1/len(sites)), draws)
        weights = np.vstack([np.ones(len(sites)), weights])
        estimates = {}
        for mode in records['mode'].unique():
            sums = np.zeros((len(sites), 15))
            for (site, station), frame in windows.groupby(['site','station']):
                x, y = frame[mode].to_numpy(), frame.target.to_numpy()
                n = len(frame)
                err = np.sqrt(np.mean((x-y)**2))
                x0, y0 = x-x.mean(), y-y.mean()
                s = sums[sites.index(site)]
                s[:7] += [1, x.mean(), y.mean(), x.mean()**2, y.mean()**2,
                          x.mean()*y.mean(), (x.mean()-y.mean())**2]
                if n >= 2:
                    s[7:14] += [n, 0, 0, x0@x0, y0@y0, x0@y0, (x0-y0)@(x0-y0)]
                s[14] += err
                stations.append(dict(task=task, mode=mode, site=site, station=station,
                                     windows=n, years=frame.date.str[:4].nunique(),
                                     start=frame.date.min(), end=frame.date.max(), rmse=err,
                                     mean_pred=x.mean(), mean_target=y.mean()))
            total = weights @ sums
            with np.errstate(invalid='ignore', divide='ignore'):
                estimates[mode] = dict(between_r=metrics(total[:,:7])[:,1],
                                       within_r=metrics(total[:,7:14])[:,1],
                                       macro_rmse=total[:,14]/total[:,0])
        counts = windows.groupby('station').date
        for mode, series in estimates.items():
            for name, values in series.items():
                delta = values-estimates['dynamic'][name]
                lo, hi = np.nanquantile(values[1:], [.025,.975])
                dlo, dhi = np.nanquantile(delta[1:], [.025,.975])
                summaries.append(dict(task=task, mode=mode, metric=name, value=values[0],
                                      low=lo, high=hi, change=delta[0], delta_low=dlo,
                                      delta_high=dhi, clusters=len(sites), stations=counts.ngroups,
                                      repeated=(counts.size()>=2).sum(), windows=len(windows),
                                      multiyear=counts.apply(lambda d: d.str[:4].nunique()>1).sum()))
    return pd.DataFrame(summaries), pd.DataFrame(stations)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('paired', type=Path)
    parser.add_argument('release', type=Path)
    parser.add_argument('--split', choices=['validation', 'test'], required=True)
    args = parser.parse_args()
    data = attach_station_identity(pd.read_parquet(args.paired), args.release, args.split)
    summary, stations = station_report(data)
    summary.to_csv(args.paired.with_name('station_summary.csv'), index=False)
    stations.to_csv(args.paired.with_name('per_station.csv'), index=False)
    print(summary.to_string(index=False))
