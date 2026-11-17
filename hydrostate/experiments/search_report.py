"""Read screening validation without combining task-wise best epochs."""

import argparse
import json
import math
from pathlib import Path

from mmengine.config import Config

TASKS = tuple(("product", task) for task in (
    "water", "snow_ice", "land", "soil_moisture", "et_water", "et_ice", "et_land")) + (
        ("site", "site_sm"), ("site", "site_et"))


def best_validation(work):
    records = []
    for path in sorted(Path(work).glob('*/vis_data/scalars.json')):
        for line in path.read_text().splitlines(keepends=True):
            if not line.endswith('\n'):
                continue  # A live logger may still be writing the final record.
            row = json.loads(line)
            if 'joint_score' in row and math.isfinite(row['joint_score']):
                records.append(row)
    if not records:
        return None
    # MMEngine greater-is-better selection keeps the first checkpoint on ties.
    best = max(records, key=lambda row: row['joint_score'])
    epoch = int(best['step'])
    checkpoint = Path(work) / f'best_joint_score_epoch_{epoch}.pth'
    return dict(epoch=epoch, completed_validation_epoch=max(int(r['step']) for r in records),
                checkpoint=str(checkpoint), checkpoint_present=checkpoint.is_file(),
                joint_score=best['joint_score'], metrics={
                    f'{source}/{task}': {metric: best[f'{source}/{task}_{metric}']
                                       for metric in ('r2', 'rmse', 'mae', 'bias', 'count')}
                    for source, task in TASKS})


def summarize(root):
    root = Path(root)
    results, contract = {}, None
    for variant in ('A', 'B', 'CGC', 'MLoRE'):
        directory = root / variant
        cfg = Config.fromfile(str(directory / 'config.py'))
        identity = dict(
            train=cfg.train_dataloader.dataset.to_dict(),
            validation=cfg.val_dataloader.dataset.to_dict(),
            statistics=cfg.model.target_statistics,
            site_statistics=cfg.model.site_supervision.statistics,
            site_weights=cfg.model.site_supervision.loss_weights.to_dict(),
            seed=cfg.randomness.seed,
            batch=cfg.train_dataloader.batch_size,
            accumulation=cfg.optim_wrapper.accumulative_counts,
            epochs=cfg.train_cfg.max_epochs)
        if contract is not None and contract != identity:
            raise ValueError('Screening candidates have different data or training budgets')
        contract = identity
        result = best_validation(directory / 'work')
        if result is not None:
            result['finished_validation_budget'] = (
                result['completed_validation_epoch'] >= cfg.train_cfg.max_epochs)
        results[variant] = result
    baseline = results['A']
    for name, result in results.items():
        if not result or not baseline or name == 'A':
            continue
        result['matched_counts'] = all(
            values['count'] == baseline['metrics'][task]['count'] and values['count'] > 0
            for task, values in result['metrics'].items())
        result['rmse_change_percent_vs_A'] = {
            task: 100 * (values['rmse'] / baseline['metrics'][task]['rmse'] - 1)
            if result['matched_counts'] and baseline['metrics'][task]['rmse'] > 0 else None
            for task, values in result['metrics'].items()}
    return dict(scope='Fixed development subset; one joint-score-selected epoch per model. '
                'Partial runs are provisional, not a final architecture ranking.', models=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', help='screen_10 directory')
    args = parser.parse_args()
    print(json.dumps(summarize(args.root), indent=2))
