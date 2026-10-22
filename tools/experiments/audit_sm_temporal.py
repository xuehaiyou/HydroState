#!/usr/bin/env python
"""Check actual common spatial support before enabling prepared C experiments."""
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd
import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope
from torch.utils.data import DataLoader
from hydrostate.experiments.sm_causal import CausalDataset, CausalPreprocessor
from hydrostate.datasets.collate import hydro_collate


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('root',type=Path)
    p.add_argument('--stage',default='screen')
    args=p.parse_args()
    init_default_scope('hydrostate')
    configs=list((args.root/args.stage).glob('C1_s*.py'))
    if not configs:
        raise ValueError('No C1 configurations')
    # Full SM pools are shared across seeds, but manifest hashes differ through
    # normalization provenance; validate each hash without repeating image I/O.
    seen={}
    config_hashes={}
    rows=[]
    for config in configs:
        cfg=Config.fromfile(config)
        manifest=Path(cfg.train_dataloader.dataset.manifest)
        config_hashes[str(manifest)]=hashlib.sha256(manifest.read_bytes()).hexdigest()
        for split in ('train','validation','external_validation'):
            base=dict(cfg.train_dataloader.dataset)
            base.pop('type')
            base.update(split=split,training=False,flip_probability=0)
            base.pop('temporal_variant',None)
            datasets={v:CausalDataset(**base,temporal_variant=v) for v in ('recent','history')}
            ids=tuple(datasets['recent'].table.sample_id)
            if (split,ids) in seen:
                continue
            seen[split,ids]=True
            loaders={v:DataLoader(d,batch_size=8,num_workers=2,collate_fn=hydro_collate,
                    multiprocessing_context='spawn') for v,d in datasets.items()}
            processor=CausalPreprocessor()
            offset=0
            for recent_raw,history_raw in zip(loaders['recent'],loaders['history']):
                recent=processor(recent_raw,False)['data_samples']
                history=processor(history_raw,False)['data_samples']
                equal=recent['prediction_valid'].eq(history['prediction_valid']).flatten(1).all(1)
                counts=recent['prediction_valid'].flatten(1).sum(1)
                labels=recent['soil_moisture_valid'].flatten(1).any(1)
                for j in range(len(counts)):
                    rows.append(dict(sample_id=ids[offset+j],split=split,
                        support_pixels=int(counts[j]),same_support=bool(equal[j]),sm_label_valid=bool(labels[j])))
                offset+=len(counts)
                if offset%128<8:
                    print(split,offset,len(ids),flush=True)
    frame=pd.DataFrame(rows)
    frame.to_csv(args.root/'temporal_support.csv',index=False)
    eligible=frame.same_support.all() and (frame.loc[frame.sm_label_valid,'support_pixels']>0).all()
    result=dict(passed=bool(eligible),component_sha256=hashlib.sha256(Path('hydrostate/experiments/sm_causal.py').read_bytes()).hexdigest(),
        manifest_sha256=config_hashes,windows=len(frame),common_support='recent causal observations, full 3x3 validity',
        cutoff='end of SM UTC product day; no assumed subdaily observation instant')
    (args.root/'temporal_support_result.json').write_text(json.dumps(result,indent=2)+'\n')
    if not eligible:
        raise RuntimeError('Common-support eligibility failed; inspect CSV and regenerate paired manifests')
    (args.root/'temporal_support_passed.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main()
