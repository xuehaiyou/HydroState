> 历史记录：不作为当前执行依据。最新入口见 [当前文档](../README.md)。

# Offline multi-task diagnostics

`tools/validation/diagnose_multitask.py` evaluates frozen checkpoints without
updating model weights or using the test split. Submit with
`scripts/legacy/slurm/13_diagnose_v2.sh` on one L40S.

The snapshot directory must contain the **expanded training config** named
`hydrostate_l40s.py`, `best.pth`, and `latest.pth`. Copy these before submission,
record source epochs and SHA256 hashes, and verify the copies. Do not point at
rolling checkpoints subject to training cleanup. The config must embed the
original target statistics and point to the original immutable manifest.

```bash
sbatch scripts/legacy/slurm/13_diagnose_v2.sh /absolute/path/to/frozen_snapshot
```

Outputs are saved under `frozen_snapshot/job_JOBID/`. The job first evaluates
the train and validation splits, then checks batch consistency and probes
gradients, for each checkpoint. The summary is generated at the end, or can be
generated from partial results with:

```bash
python tools/validation/summarize_multitask_diagnostics.py /path/to/job_JOBID
```

For an explicitly limited preflight:

```bash
sbatch --time=01:00:00 \
  --export=ALL,MAX_WINDOWS=256,CONSISTENCY_WINDOWS=16,GRADIENT_BATCHES_PER_SEED=4 \
  scripts/legacy/slurm/13_diagnose_v2.sh /absolute/path/to/frozen_snapshot
```

## Outputs and interpretation

- `*_windows.csv`: per-task/per-window sufficient statistics on actual valid
  observational support, coordinates, climate, month and sensor availability.
- `*_metrics.json`: observation-weighted metrics, equal-window MSE-root,
  valid-window counts and 1,000-replicate spatial-block bootstrap RMSE intervals.
- `*_groups.csv`: climate, month, modality and 5-degree geographic block metrics.
- `batch_consistency.csv`: same-window prediction differences between singleton,
  repeated singleton, batch 2/16/32 and reordered batch 16 inference.
- `gradients.csv`: MSE and current NLL gradient norms, pairwise cosines and dot
  products with the combined gradient. Three seeds, 34 global batches per seed
  by default. Gradients cover the shared decoder and last encoder block only.
- `sampler_batches.csv`: five-epoch replacement-sampling replay using actual
  final validity, with task-bearing window counts per global batch.
- `completed.json`: all requested stages finished. Its absence means partial
  results, even if some metric files exist.

The diagnostic uses FP32 eval mode. Its single-GPU batch 32 gradient forward is
not the same as two separate DDP forwards at batch 16 if the model is batch
dependent. NLL shared-parameter gradients are obtained exactly from MSE
gradients via `1/(2*variance)` for the current Gaussian loss with no station
labels. This is not an AdamW optimizer-step replay. Zero-norm pairwise cosines
are undefined and stored as missing.

Preflight sampling metrics refer only to the sampled subset. Full-run task
coverage is not inferred from preflight. Neither stage counts independent
native product pixel/date IDs; this requires a separate provenance audit.
Spatial block bootstrap accounts for some spatial dependence but does not make
windows independent observations. Batch-dependent predictions must be resolved
before interpreting sampling/optimization experiments as causal evidence.

Current snapshot: `outputs/diagnostics_v2_4107932/frozen_epoch32`, with epoch 20
best and epoch 32 latest at collection time. Later training progress does not
change that snapshot. Detailed experimental design is in
`outputs/diagnostics_v2_4107932/PLAN.md`.
