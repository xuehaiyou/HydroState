"""Summarize completed or partial offline diagnostics without hiding missing stages."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    root = args.output
    tasks = ('water', 'soil_moisture', 'et')
    lines = ['# v2 多任务诊断结果', '',
             f'状态：{"全部诊断阶段完成" if (root / "completed.json").exists() else "部分结果；尚未全部完成"}。', '',
             '范围：作业 4107932 的冻结检查点 best=epoch 20、latest=epoch 32；不评估测试集。',
             '推理统一 FP32、eval、batch=16，关闭增强；不能直接与原 DDP 验证数值混合。',
             '误差单位：WF 为比例，SM 为 m³/m³，ET 为 mm/day。', '']
    structured = {}
    conditions_path = root / 'conditions.json'
    if conditions_path.exists():
        conditions = json.loads(conditions_path.read_text())
        if conditions['max_windows']:
            lines.extend(['**抽样预检：不是全量结果，采样覆盖和误差仅代表抽取的子集。**', '',
                          f"各 split 窗口数：{conditions['split_windows']}", ''])
        if conditions['consistency_windows'] < 32:
            lines.extend([f"一致性预检最多 {conditions['consistency_windows']} 个窗口；batch32 是请求的上限，实际批次不会超过子集大小。", ''])
    for name in ('best', 'latest'):
        directory = root / name
        if not directory.exists():
            continue
        lines.extend([f'## {name}', '', '| 任务 | 训练 RMSE | 验证 RMSE | 验证/训练 | 有效训练/验证窗口 | 验证 bias |',
                      '|---|---:|---:|---:|---:|---:|'])
        data = {}
        for split in ('train', 'validation'):
            path = directory / f'{split}_metrics.json'
            if path.exists():
                data[split] = json.loads(path.read_text())
        structured[name] = data
        if len(data) == 2:
            for task in tasks:
                a, b = data['train'][task], data['validation'][task]
                ratio = b['rmse']/a['rmse'] if a['rmse'] else float('inf')
                lines.append(f"| {task} | {a['rmse']:.5f} | {b['rmse']:.5f} | {ratio:.2f} | {a['valid_windows']}/{b['valid_windows']} | {b['bias']:.5f} |")
        else:
            lines.append('完整训练/验证评估尚未结束。')
        path = directory / 'batch_consistency.csv'
        if path.exists():
            df = pd.read_csv(path)
            table = df.groupby(['variant', 'task']).agg(max_abs=('max_abs', 'max'), mean_window_rmse=('rmse', 'mean'))
            lines.extend(['', '### 批次一致性', '', '各批次方案相对逐样本推理的最大绝对差：', '',
                          '| 方案 | WF | SM | ET |', '|---|---:|---:|---:|'])
            for variant in df.variant.unique():
                vals = [table.loc[(variant, t), 'max_abs'] if (variant, t) in table.index else np.nan for t in tasks]
                lines.append(f'| {variant} | ' + ' | '.join(f'{v:.6g}' for v in vals) + ' |')
            table.to_csv(directory / 'batch_consistency_summary.csv')
            repeat = df[df.variant == 'repeat'].max_abs.max()
            cross = df[df.variant != 'repeat'].max_abs.max()
            if cross > max(1e-5, repeat*10):
                lines.extend(['', '**存在超出重复推理噪声的批次依赖：先定位前向传播，再开展采样/优化因果对照。**',
                              '这里比较同一 checkpoint、同一样本的预测变化，不是不同样本间误差。'])
            windows = pd.read_csv(directory / 'validation_windows.csv', dtype={'modalities': str})
            joined = df.merge(windows[['index', 'task', 'modalities']], on=['index', 'task'])
            grouped = joined[joined.variant == 'batch16'].groupby(['modalities', 'task']).agg(
                windows=('index', 'size'), max_abs=('max_abs', 'max'))
            grouped.to_csv(directory / 'consistency_by_modality.csv')
        path = directory / 'gradients.csv'
        if path.exists():
            df = pd.read_csv(path)
            df = df[(df.valid_count > 0) & (df.norm > 0)]
            lines.extend(['', '### 梯度探针', '',
                          '仅测共享解码器与编码器末层；FP32 eval，合成单卡 batch=32，非 AdamW 实际更新。', '',
                          '| 区域 | 损失 | 任务 | 探针数 | 范数中位数 | 合成方向一阶不利比例 |',
                          '|---|---|---|---:|---:|---:|'])
            for (scope, objective, task), g in df.groupby(['scope', 'objective', 'task']):
                lines.append(f'| {scope} | {objective} | {task} | {len(g)} | {g.norm.median():.5g} | {(g.dot_combined<0).mean():.1%} |')
            lines.extend(['', '| 区域（NLL） | 任务对 | 有效探针数 | cosine 中位数 | 负 cosine 比例 |',
                          '|---|---|---:|---:|---:|'])
            for scope in df.scope.unique():
                for i, task in enumerate(tasks):
                    for other in tasks[i+1:]:
                        values = df.loc[(df.scope == scope) & (df.objective == 'nll') & (df.task == task), f'cos_{other}'].dropna()
                        if len(values):
                            lines.append(f'| {scope} | {task}/{other} | {len(values)} | {values.median():.4f} | {(values<0).mean():.1%} |')
        path = directory / 'sampler_batches.csv'
        if path.exists():
            f = pd.read_csv(path)
            lines.extend(['', f"均匀有放回采样重放：SM 全局 batch 缺席率 {(f.soil_moisture == 0).mean():.2%}，每 batch 有效 SM 窗口均值 {f.soil_moisture.mean():.2f}。"])
    if all(len(structured.get(n, {})) == 2 for n in ('best', 'latest')):
        lines.extend(['', '## 第 20→32 轮变化', '', '| 任务 | 训练 RMSE 变化 | 验证 RMSE 变化 |', '|---|---:|---:|'])
        for task in tasks:
            values = [structured['latest'][s][task]['rmse']/structured['best'][s][task]['rmse']-1 for s in ('train', 'validation')]
            lines.append(f'| {task} | {values[0]:+.1%} | {values[1]:+.1%} |')
    lines.extend(['', '## 解释边界', '',
                  '- 原始梯度的方向冲突不能直接证明实际 AdamW 更新或验证泛化受损。',
                  '- 若存在系统性批次依赖，应先修复并复核，再解释采样或优化算法对照。',
                  '- 空间块置信区间及分组结果见 *_metrics.json 和 *_groups.csv；原生观测去重仍需额外审计。',
                  '- 单任务及 Nash-MTL 因果对照尚未执行，不能宣称任何算法已被验证最优。', ''])
    (root / 'report.md').write_text('\n'.join(lines))
    print(root / 'report.md')


if __name__ == '__main__':
    main()
