"""Reserve whole station groups inside the architecture screening train split."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd
from mmengine.config import Config

from hydrostate.data.site.archives import sha256
from hydrostate.data.site.statistics import cohort_statistics


def heldout_groups(frames):
    parent = {}

    def root(key):
        parent.setdefault(key, key)
        while key != parent[key]:
            key = parent[key]
        return key

    for table in frames:
        for spatial, physical in table[['spatial_group_id', 'physical_group_id']].drop_duplicates().itertuples(index=False, name=None):
            parent[root(('p', physical))] = root(('s', spatial))
    groups = [table.spatial_group_id.map(lambda key: root(('s', key))) for table in frames]
    chosen = set()
    for ids in groups:
        sizes = ids.value_counts()
        if len(sizes) < 2:
            continue
        ranked = sorted(sizes.index, key=lambda x: hashlib.sha256(repr(x).encode()).digest())
        quota = max(1, round(len(ids) * .1))
        for group in ranked:
            if ids.isin(chosen).sum() >= quota:
                break
            if all(len(set(other) - chosen - {group}) > 0 for other in groups):
                chosen.add(group)
    return [ids.isin(chosen) for ids in groups]


def prepare(config, output):
    cfg = Config.fromfile(config)
    options = cfg.train_dataloader.dataset
    source = Path(options.selection_root)
    split = Path(options.site_split_root)
    definitions = json.loads((split / 'datasets.json').read_text())['datasets']
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'ready.json').exists():
        return
    selection = out / 'selection'
    shutil.copytree(source, selection, dirs_exist_ok=True)
    audit = json.loads((selection / 'audit.json').read_text())
    records = [audit['splits']['train'][str(i)]['subsets']['0.1']
               for i in range(1, len(definitions) + 1)]
    tables = [pd.read_parquet(selection / r['manifest']) for r in records]
    plans, fit, meta = [], [], []
    for i, (item, r, table, hold) in enumerate(zip(definitions, records, tables, heldout_groups(tables))):
        path = selection / r['manifest']
        table.loc[~hold].to_parquet(path, index=False)
        r['sha256'] = sha256(path)
        data_root = str(split / item['data_root'])
        fit.append(dict(manifest=str(path), data_root=data_root))
        if hold.any():
            path = out / f'meta_{i}.parquet'
            table.loc[hold].to_parquet(path, index=False)
            meta.append(dict(manifest=str(path), data_root=data_root, task=item['task']))
        plans.append(dict(task=item['task'], fit=int((~hold).sum()), meta=int(hold.sum())))
    assert {m['task'] for m in meta} == {'sm','et'}
    (selection / 'audit.json').write_text(json.dumps(audit))
    (out / 'site_stats.json').write_text(json.dumps(cohort_statistics(fit)))
    (out / 'ready.json').write_text(json.dumps(dict(meta=meta, cohorts=plans),indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('output')
    prepare(**vars(parser.parse_args()))
