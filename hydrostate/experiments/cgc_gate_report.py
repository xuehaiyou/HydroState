"""Paired changes in station metrics with uncertainty clustered by spatial group."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def metrics(sums):
    n, sx, sy, sxx, syy, sxy, se2 = sums.T
    variance = (sxx-sx*sx/n).clip(0) * (syy-sy*sy/n).clip(0)
    corr = np.divide(sxy-sx*sy/n, np.sqrt(variance),
                     out=np.full_like(n, np.nan), where=variance > 0)
    return np.stack((np.sqrt(se2/n), corr), axis=-1)


def compare(table, draws=2000, seed=20261121):
    keys = ['site','sample_id','observation']
    output = []
    for task, rows in table.groupby('task'):
        paired = rows.pivot(index=keys, columns='mode', values='pred')
        if paired.isna().any().any() or 'dynamic' not in paired:
            raise ValueError('Each intervention must predict every reference observation')
        labels = rows.groupby(keys).target
        if (labels.nunique()!=1).any():
            raise ValueError('Labels differ across interventions')
        y = labels.first().reindex(paired.index).to_numpy()
        sites = pd.factorize(paired.index.get_level_values('site'))[0]
        k = sites.max()+1
        if k<2:
            raise ValueError('Multiple station groups required')
        generator = np.random.default_rng(seed)
        # Each bootstrap draw samples whole spatial groups, identically for all modes.
        weights = generator.multinomial(k, np.full(k,1/k), size=draws)
        def compute(x):
            raw = np.stack((np.ones(len(y)),x,y,x*x,y*y,x*y,(x-y)**2),axis=-1)
            sums = np.zeros((k,7))
            np.add.at(sums,sites,raw)
            return metrics(sums.sum(0, keepdims=True))[0], metrics(weights @ sums)
        baseline, ref_draws = compute(paired.dynamic.to_numpy())
        for mode in paired:
            values, replicates = compute(paired[mode].to_numpy())
            for i, metric in enumerate(('rmse','pearson_r')):
                lo,hi = np.nanquantile(replicates[:,i]-ref_draws[:,i], [.025,.975])
                output.append(dict(task=task, mode=mode, metric=metric,
                    value=values[i], change=values[i]-baseline[i], lower=lo, upper=hi,
                    windows=paired.index.get_level_values('sample_id').nunique(),
                    observations=len(y), spatial_groups=k))
    return pd.DataFrame(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('paired', type=Path)
    args = parser.parse_args()
    result = compare(pd.read_parquet(args.paired))
    result.to_csv(args.paired.with_name('gate_effects.csv'), index=False)
    print(result.to_string(index=False))
