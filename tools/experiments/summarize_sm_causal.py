#!/usr/bin/env python
"""Paired spatial-block uncertainty for completed A-stage comparisons."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def paired(left,right):
    a=left[(left.task=='soil_moisture')&(left['count']>0)].set_index('sample_id')
    b=right[(right.task=='soil_moisture')&(right['count']>0)].set_index('sample_id')
    if set(a.index)!=set(b.index):
        raise ValueError('Paired evaluation supports differ')
    b=b.reindex(a.index)
    if not np.array_equal(a['count'],b['count']) or not np.allclose(a.y,b.y):
        raise ValueError('Paired labels or observation counts differ')
    frame=pd.DataFrame({'block':a.block,'n':a['count'],'a':a.sq,'b':b.sq})
    blocks=frame.groupby('block')[['n','a','b']].sum().to_numpy()
    rng=np.random.default_rng(20260925)
    diffs=[]
    for _ in range(1000):
        n,x,y=blocks[rng.integers(len(blocks),size=len(blocks))].sum(axis=0)
        diffs.append(float(np.sqrt(y/n)-np.sqrt(x/n)))
    n,x,y=blocks.sum(axis=0)
    return dict(reference_rmse=float(np.sqrt(x/n)),candidate_rmse=float(np.sqrt(y/n)),
                candidate_minus_reference=float(np.sqrt(y/n)-np.sqrt(x/n)),
                delta_ci95=np.quantile(diffs,[.025,.975]).tolist(),spatial_blocks=len(blocks),
                relative_improvement=float(1-np.sqrt(y/x)))


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('root',type=Path)
    args=p.parse_args()
    matrix=pd.read_csv(args.root/'screen/run_matrix.csv')
    reports={}
    rows=[]
    for r in matrix.itertuples():
        work=args.root/'screen/runs'/r.run_id
        if not (work/'evaluation/completed.json').exists():
            continue
        reports[r.run_id]=json.loads((work/'evaluation/completed.json').read_text())
        for checkpoint,splits in reports[r.run_id]['checkpoints'].items():
            for split,metrics in splits.items():
                sm=metrics['soil_moisture']
                rows.append(dict(run_id=r.run_id,checkpoint=checkpoint,split=split,
                    rmse=sm['rmse'],mae=sm['mae'],bias=sm['bias'],r2=sm['r2'],windows=sm['valid_windows']))
    comparisons={}
    for seed in matrix.seed.unique():
        for reference,candidate in [('A0','A1'),('A2','A3'),('A0','A2')]:
            a,b=f'{reference}_s{seed}',f'{candidate}_s{seed}'
            if a not in reports or b not in reports:
                continue
            for checkpoint in ('best','last'):
                paths=[args.root/'screen/runs'/r/'evaluation'/checkpoint/'external_validation_windows.csv' for r in (a,b)]
                comparisons[f'{a}_vs_{b}_{checkpoint}']=paired(*(pd.read_csv(p) for p in paths))
    output=args.root/'screen/summary'
    output.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(output/'metrics.csv',index=False)
    (output/'paired_comparisons.json').write_text(json.dumps(comparisons,indent=2)+'\n')
    text=['# SM 对照实验筛查结果','',
          '这是单种子短程筛查；不能替代完整预算、多种子的因果确认。',
          'best 由原训练集内部空间留出选择；外部验证未参与检查点选择；未评估测试集。','',
          '| 对照 | 参考 RMSE | 候选 RMSE | 候选−参考 95% CI |','|---|---:|---:|---|']
    for name,c in comparisons.items():
        text.append(f"| {name} | {c['reference_rmse']:.5f} | {c['candidate_rmse']:.5f} | {c['delta_ci95']} |")
    text+=['',f'已完成评估：{len(reports)} 组。其余状态见 run_matrix.csv 和 Slurm 作业。',
           '负差值支持候选更好；区间为固定模型的空间块重采样，不包含跨训练种子不确定性。',
           '仅激活 SM 的模型，其 WF/ET 输出未训练，不用于多任务精度结论。']
    (output/'report.md').write_text('\n'.join(text)+'\n')


if __name__=='__main__':
    main()
