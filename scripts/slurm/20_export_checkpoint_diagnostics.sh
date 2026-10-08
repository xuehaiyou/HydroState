#!/bin/bash
#SBATCH --job-name=hs_diagnostics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=l40s
#SBATCH --account=geog_geors
#SBATCH --qos=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --time=01:00:00
#SBATCH --mail-type=NONE
#SBATCH --output=/home/xiaoz/HydroState/logs/%x-%j.out
#SBATCH --error=/home/xiaoz/HydroState/logs/%x-%j.err
set -euo pipefail
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    cat <<'HELP'
Usage: sbatch scripts/slurm/20_export_checkpoint_diagnostics.sh [RUN_DIR] [SEED]
Default RUN_DIR: HYDROSTATE_WORK_DIR, otherwise newest nll_*/v2_nll_* last_checkpoint.
Uses last_checkpoint (latest, not best), saved config, and the entire validation split.
Outputs: scatter.png/pdf, metrics.json/csv, scatter_points.npz, 3 random distinct
 tile comparisons (PNG), and provenance.json under RUN_DIR/checkpoint_diagnostics/CKPT/seed_SEED.
Overrides: HYDROSTATE_EXPORT_CHECKPOINT, HYDROSTATE_EXPORT_CONFIG,
 HYDROSTATE_EXPORT_DIR (relative to RUN_DIR), HYDROSTATE_SCATTER_MAX_POINTS (50000/task),
 HYDROSTATE_VIS_ET_VMAX (10 mm/day), HYDROSTATE_RUN_ROOT.
Metrics use all valid observations; only displayed scatter points are subsampled.
Tile selection is uniform over distinct tiles with valid WF, SM AND ET labels,
independent of scores. One eligible validation date is randomly selected per tile.
Each task must have at least one valid observation on that date. SM labels are
scalars; ET maps are footprint projections. Metrics still use all validation data.
HELP
    exit 0
fi
[[ $# -le 2 ]] || { echo 'Expected [RUN_DIR] [SEED]' >&2; exit 2; }
run_dir="${1:-${HYDROSTATE_WORK_DIR:-}}"
[[ -z "$run_dir" ]] || run_dir="$(cd "$run_dir" && pwd -P)"
seed="${2:-42}"
[[ "$seed" =~ ^[0-9]+$ ]] || { echo 'SEED must be nonnegative' >&2; exit 2; }
cd "${HYDROSTATE_CODE_ROOT:-/home/xiaoz/HydroState}"
source "${HYDROSTATE_CONDA_BASE:-/home/xiaoz/miniconda}/etc/profile.d/conda.sh"
conda activate "${HYDROSTATE_CONDA_ENV:-hydrostate}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python - "$run_dir" "$seed" <<'PY'
import csv
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mmengine.config import Config
from mmengine.hooks import Hook
from mmengine.runner import Runner
from hydrostate.cli import configure_target_statistics
from hydrostate.engine.visualization_hook import HydroVisualizationHook, _cpu_sample
from hydrostate.normalization import TASKS
from hydrostate.registry import HOOKS

seed = int(sys.argv[2])
if sys.argv[1]:
    run = Path(sys.argv[1])
else:
    root = Path(os.environ.get('HYDROSTATE_RUN_ROOT', '/scr/u/xiaoz/HydroState/work_dirs'))
    pointers = [p for pattern in ('nll_*/last_checkpoint', 'v2_nll_*/last_checkpoint')
                for p in root.glob(pattern)]
    if not pointers:
        raise FileNotFoundError(f'No training last_checkpoint in {root}; supply RUN_DIR')
    run = max(pointers, key=lambda p: (p.stat().st_mtime_ns, str(p))).parent

def resolve(value):
    p = Path(value).expanduser()
    return p if p.is_absolute() else run / p

name = os.environ.get('HYDROSTATE_EXPORT_CHECKPOINT') or (run / 'last_checkpoint').read_text().strip()
if not name:
    raise ValueError('Empty last_checkpoint')
checkpoint = resolve(name)
if not checkpoint.is_file():
    raise FileNotFoundError(checkpoint)
config_name = os.environ.get('HYDROSTATE_EXPORT_CONFIG')
configs = [resolve(config_name)] if config_name else sorted(run.glob('*.py'))
if len(configs) != 1:
    raise ValueError('Set HYDROSTATE_EXPORT_CONFIG: expected exactly one saved config')
cfg = Config.fromfile(str(configs[0]))
configure_target_statistics(cfg, training=False)
if cfg.get('val_dataloader') is None or cfg.get('val_evaluator') is None:
    raise ValueError('Saved config must contain validation data and evaluator')
# Some manifests omit tile_id; derive a stable spatial footprint ID instead.
dataset_cfg = cfg.val_dataloader.dataset
manifest = pd.read_parquet(dataset_cfg.manifest)
manifest = manifest.loc[manifest['split'] == dataset_cfg.get('split', 'validation')]
tile_lookup = {}
for _, row in manifest.iterrows():
    tile = row.get('tile_id')
    if tile is None or pd.isna(tile) or not str(tile).strip():
        if not all(k in row and pd.notna(row[k]) for k in ('lon', 'lat', 'crs')):
            raise ValueError('Missing tile_id and spatial coordinates in manifest')
        footprint = json.dumps([str(row['crs']), float(row['lon']), float(row['lat']),
                                str(row.get('transform', ''))])
        tile = 'spatial_' + hashlib.sha256(footprint.encode()).hexdigest()[:20]
    tile_lookup[str(row['sample_id'])] = str(tile)
if len(set(tile_lookup.values())) < 3:
    raise ValueError('Validation manifest contains fewer than 3 distinct spatial tiles')
print(f'Validation spatial tiles: {len(set(tile_lookup.values()))}', flush=True)
output = resolve(os.environ.get('HYDROSTATE_EXPORT_DIR') or
                 f'checkpoint_diagnostics/{checkpoint.stem}/seed_{seed}')
output.mkdir(parents=True, exist_ok=True)
limit = int(os.environ.get('HYDROSTATE_SCATTER_MAX_POINTS', '50000'))
if limit < 1:
    raise ValueError('HYDROSTATE_SCATTER_MAX_POINTS must be positive')

@HOOKS.register_module()
class CheckpointDiagnosticsHook(Hook):
    def before_val_epoch(self, runner):
        self.rng = np.random.default_rng(seed)
        self.points = {t: np.empty((0, 3)) for t in TASKS}
        self.counts = {t: 0 for t in TASKS}
        self.tiles = {}
        self.seen_tiles = set()
        self.eligible_tiles = set()
        self.eligible_samples = 0
        self.seen_samples = set()

    def sample_priority(self, kind, value):
        return hashlib.sha256(f'{seed}:{kind}:{value}'.encode()).hexdigest()

    def after_val_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        for raw in outputs:
            sample = _cpu_sample(raw)
            meta = sample['metainfo']
            sid = str(meta['sample_id'])
            if sid in self.seen_samples:
                raise ValueError(f'Duplicate validation sample: {sid}')
            self.seen_samples.add(sid)
            for task in TASKS:
                y = np.asarray(sample[f'target_{task}'], dtype=np.float64)
                p = np.asarray(sample[f'supervised_{task}'], dtype=np.float64)
                mask = np.broadcast_to(sample[f'valid_{task}'].astype(bool), y.shape)
                y, p = y[mask], p[mask]
                if not (np.isfinite(y).all() and np.isfinite(p).all()):
                    raise ValueError(f'Nonfinite valid {task} observations in {sid}')
                self.counts[task] += len(y)
                # Independent random priorities give a bounded uniform point sample.
                rows = np.column_stack((self.rng.random(len(y)), y, p))
                rows = np.concatenate((self.points[task], rows))
                if len(rows) > limit:
                    rows = rows[np.argpartition(rows[:, 0], limit - 1)[:limit]]
                self.points[task] = rows
            tile = tile_lookup[sid]
            meta['tile_id'] = tile
            self.seen_tiles.add(tile)
            # Require all three labels on the SAME date, after input-coverage masking.
            # This filter affects only tile images; all observations above remain
            # in the scatter sample and the unchanged validation evaluator.
            if not all(np.asarray(sample[f'valid_{task}']).any() for task in TASKS):
                continue
            self.eligible_tiles.add(tile)
            self.eligible_samples += 1
            key = (self.sample_priority('tile', tile), tile)
            date_key = (self.sample_priority('sample', sid), sid)
            if tile in self.tiles:
                if date_key < self.tiles[tile][1]:
                    self.tiles[tile] = (key, date_key, sample)
            elif len(self.tiles) < 3:
                self.tiles[tile] = (key, date_key, sample)
            else:
                worst = max(self.tiles, key=lambda t: self.tiles[t][0])
                if key < self.tiles[worst][0]:
                    del self.tiles[worst]
                    self.tiles[tile] = (key, date_key, sample)

    def after_val_epoch(self, runner, metrics=None):
        if len(self.tiles) < 3:
            raise ValueError(f'Need 3 distinct tiles with valid WF/SM/ET on the same date; found {len(self.tiles)}')
        clean = {k: float(v) if np.isfinite(v) else None for k, v in metrics.items()}
        (output / 'metrics.json').write_text(json.dumps(clean, indent=2, allow_nan=False) + '\n')
        with (output / 'metrics.csv').open('w') as f:
            writer = csv.writer(f)
            writer.writerow(['task', 'n', 'r2', 'rmse', 'mae', 'bias'])
            for t in TASKS:
                writer.writerow([t, self.counts[t]] + [clean.get(f'{t}_{m}') for m in ('r2', 'rmse', 'mae', 'bias')])
        plt.rcParams.update({'pdf.fonttype': 42, 'font.size': 10})
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), layout='constrained')
        for ax, task, title in zip(axes, TASKS, ('WF (fraction)', 'SM (m³/m³)', 'ET (mm/day)')):
            points = self.points[task][:, 1:]
            if len(points):
                ax.scatter(points[:, 0], points[:, 1], s=3, alpha=.18, linewidths=0, rasterized=True)
                lo, hi = float(points.min()), float(points.max())
                pad = max((hi - lo) * .04, 1e-6)
                ax.plot([lo-pad, hi+pad], [lo-pad, hi+pad], '--', color='0.3', lw=1)
                ax.set(xlim=(lo-pad, hi+pad), ylim=(lo-pad, hi+pad))
            text = '\n'.join(f'{label} = {metrics.get(f"{task}_{m}", float("nan")):.4f}'
                             for label, m in [('R²', 'r2'), ('RMSE', 'rmse'), ('MAE', 'mae'), ('Bias', 'bias')])
            ax.text(.04, .96, f'{text}\nN = {self.counts[task]:,}\nShown = {len(points):,}',
                    transform=ax.transAxes, va='top', bbox=dict(facecolor='white', alpha=.85, edgecolor='none'))
            ax.set(title=title, xlabel='Label (supervision scale)', ylabel='Prediction (supervision scale)')
            ax.set_aspect('equal', adjustable='box')
        fig.suptitle(f'{checkpoint.name} | validation | metrics use all valid observations')
        fig.savefig(output / 'scatter.png', dpi=220)
        fig.savefig(output / 'scatter.pdf')
        plt.close(fig)
        np.savez_compressed(output / 'scatter_points.npz', **{t: p[:, 1:] for t, p in self.points.items()})
        drawer = HydroVisualizationHook(num_samples=3, et_vmax=float(os.environ.get('HYDROSTATE_VIS_ET_VMAX', '10')))
        drawer._label = f'{checkpoint.stem} | random seed {seed}'
        selected = []
        for i, (_, _, sample) in enumerate(sorted(self.tiles.values(), key=lambda x: x[0]), 1):
            meta = sample['metainfo']
            filename = f'tile_{i:02d}_{re.sub(r"[^A-Za-z0-9_.-]", "_", str(meta["tile_id"]))[:80]}.png'
            drawer._draw(sample, output / filename)
            selected.append(dict(file=filename, metainfo=meta,
                                 valid_counts={t: int(np.asarray(sample[f'valid_{t}']).sum()) for t in TASKS}))
        provenance = dict(checkpoint=str(checkpoint), config=str(configs[0]), seed=seed,
                          validation_samples=len(self.seen_samples), validation_tiles=len(self.seen_tiles),
                          eligible_samples=self.eligible_samples, eligible_tiles=len(self.eligible_tiles),
                          counts=self.counts, scatter_columns=['label', 'prediction'],
                          selection='Uniform seeded hash priorities over tiles with all three valid labels, then eligible sample IDs',
                          eligibility='At least one valid supervision observation per task on the same sample/date',
                          metrics_scope='All valid supervision-scale observations, original physical units',
                          selected=selected)
        (output / 'provenance.json').write_text(json.dumps(provenance, indent=2, default=str) + '\n')
        print(f'Diagnostics saved to {output}', flush=True)

cfg.work_dir = str(output)
cfg.load_from = str(checkpoint)
cfg.resume = False
cfg.launcher = 'none'
for key in ('train_cfg', 'train_dataloader', 'optim_wrapper', 'param_scheduler'):
    cfg[key] = None
cfg.default_hooks = dict(cfg.get('default_hooks') or {}, checkpoint=None)
cfg.log_processor = dict(cfg.get('log_processor') or {}, by_epoch=True)
cfg.custom_hooks = [dict(type='CheckpointDiagnosticsHook')]
print(f'Config: {configs[0]}\nCheckpoint: {checkpoint}\nOutput: {output}', flush=True)
Runner.from_cfg(cfg).val()
PY
