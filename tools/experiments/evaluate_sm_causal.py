#!/usr/bin/env python
"""Evaluate best-by-inner-dev and last models on train/dev/external validation."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint
import hydrostate.experiments.sm_causal  # noqa
from hydrostate.registry import DATASETS, MODELS
from tools.validation.diagnose_multitask import evaluate


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('config',type=Path)
    args=p.parse_args()
    cfg=Config.fromfile(args.config)
    work=Path(cfg.work_dir)
    done=json.loads((work/'completed.json').read_text())
    if done['smoke'] or done['iterations'] != cfg.train_cfg.max_iters:
        raise RuntimeError('Training is not complete')
    current=hashlib.sha256(Path('hydrostate/experiments/sm_causal.py').read_bytes()).hexdigest()
    if current != done['component_sha256']:
        raise RuntimeError('Evaluation code differs from training')
    init_default_scope('hydrostate')
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(cfg.randomness.seed)
    best=list(work.glob('best_soil_moisture_rmse*.pth'))
    if len(best)!=1:
        raise RuntimeError(f'Expected one selected checkpoint, got {best}')
    checkpoints={'best':best[0],'last':Path((work/'last_checkpoint').read_text().strip())}
    model=MODELS.build(cfg.model).cuda().eval()
    report={'active_tasks':list(cfg.model.active_tasks),'selection':'inner spatial-block dev',
            'screening_only':cfg.causal_protocol.screening_only,'test_evaluated':False,'checkpoints':{}}
    for label,checkpoint in checkpoints.items():
        load_checkpoint(model,str(checkpoint),map_location='cpu',strict=True)
        model.eval()
        output=work/'evaluation'/label
        output.mkdir(parents=True,exist_ok=True)
        report['checkpoints'][label]={}
        for split in ('train','validation','external_validation'):
            dc=dict(cfg.train_dataloader.dataset)
            dc.update(split=split,training=False,flip_probability=0)
            ds=DATASETS.build(dc)
            _,metrics=evaluate(model,ds,split,output,batch=16,workers=2)
            report['checkpoints'][label][split]=metrics
    (work/'evaluation/completed.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
