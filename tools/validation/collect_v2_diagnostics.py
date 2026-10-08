"""Read-only run/manifest inventory; no inference or test-label evaluation."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    table = pd.read_parquet(args.manifest)
    histories = sorted(args.run.glob('*/vis_data/scalars.json'))
    if len(histories) != 1:
        raise ValueError(f'Expected one history, found {histories}')
    raw = histories[0].read_bytes()
    # A running writer may leave a partial last line. Preserve only complete lines.
    complete = raw[:raw.rfind(b'\n') + 1]
    rows = [json.loads(line) for line in complete.splitlines() if line.strip()]
    val = [r for r in rows if 'hydro_score' in r]
    train = [r for r in rows if 'loss' in r]
    tasks = ('water', 'soil_moisture', 'et')
    metadata = {}
    for split in ('train', 'validation'):
        frame = table.loc[table.split == split].copy()
        labels = frame.labels.map(json.loads)
        inputs = frame.inputs.map(json.loads)
        availability = labels.map(lambda x: ''.join(
            name for name, key in [('W', 'dw'), ('S', 'sm'), ('E', 'pml')]
            if x.get(key) is not None) or 'none')
        metadata[split] = {
            'windows': len(frame),
            'label_metadata_patterns_NOT_final_validity': availability.value_counts().to_dict(),
            's1_metadata_available_windows': int(inputs.map(
                lambda x: any(v is not None for k, v in x.items() if k.startswith('s1_'))).sum()),
            'climate_class_windows': frame.climate_class.value_counts(dropna=False).to_dict(),
            'months': pd.to_datetime(frame.target_time).dt.month.value_counts().to_dict(),
            'sample_weight': frame.sample_weight.describe().to_dict()
                if 'sample_weight' in frame else 'absent: uniform replacement sampling',
        }
    summary = {
        'collected_utc': datetime.now(timezone.utc).isoformat(),
        'run': str(args.run), 'manifest': str(args.manifest),
        'manifest_sha256': hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        'history_sha256': hashlib.sha256(complete).hexdigest(),
        'split_counts': table.split.value_counts().to_dict(),
        'duplicate_sample_ids': int(table.sample_id.duplicated().sum()),
        'training_statistics': json.loads(args.manifest.with_name('target_stats.json').read_text()),
        'metadata_audit': metadata,
        'latest_validation': val[-1] if val else None,
        'best_hydro_score': max(val, key=lambda r: r['hydro_score']) if val else None,
        'best_task_rmse': {t: min(val, key=lambda r: r[f'{t}_rmse']) for t in tasks} if val else {},
        'latest_logged_training': train[-1] if train else None,
        'available_checkpoints': [{'path': str(p), 'bytes': p.stat().st_size}
                                  for p in sorted(args.run.glob('*.pth'))],
        'limitations': ['Metadata presence is not final supervision validity.',
                       'Logged training values are rolling summaries, not full-split metrics.',
                       'Historical checkpoints may already have been pruned.',
                       'Test labels and test predictions were not evaluated.'],
    }
    (args.out / 'history_snapshot.jsonl').write_bytes(complete)
    pd.DataFrame(val).to_csv(args.out / 'validation_history.csv', index=False)
    (args.out / 'inventory.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: summary[k] for k in ('split_counts', 'latest_validation', 'best_hydro_score')}, indent=2))


if __name__ == '__main__':
    main()
