# SM paired experiments

Protocol: [SM_CAUSAL_EXPERIMENT_PLAN.md](../../docs/SM_CAUSAL_EXPERIMENT_PLAN.md).
Components are opt-in under `hydrostate.experiments.sm_causal`; existing training
configurations and the running v2 job do not import them.

Initial submission is four **single-seed, 2340-update screening** runs:

| Array index | Run | Tasks | Loss |
|---|---|---|---|
| 0 | A0 | WF + SM + ET | learned NLL |
| 1 | A1 | SM | learned NLL |
| 2 | A2 | WF + SM + ET | standardized MSE |
| 3 | A3 | SM | standardized MSE |

All use two GPUs, batch 16 each, identical saved global batch rows and paired
initializations. Global batches lacking SM are rejected **for all four arms**;
the changed sampler is explicitly recorded, not passed off as the original run.
Inner spatial holdout selects checkpoints. External validation never selects
checkpoints and the test set is not evaluated. Checkpoint selection prioritizes SM
for every arm. Both selected and last checkpoints are evaluated after training.

The isolated encoder groups identical modality/aligned-time-mask layouts before
calling the installed OlmoEarth wrapper. This avoids dependence on unrelated
batch neighbors without editing site-packages or the original adapter. It may
cost throughput; GPU comparisons to singleton inference and backward smoke are
mandatory. It is not an upstream root-cause fix for every padding behavior.

For a fresh output directory, from the repository root:

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
PY=/home/xiaoz/miniconda/envs/hydrostate/bin/python
ROOT=/home/xiaoz/HydroState/outputs/sm_causal_NEW
mkdir -p "$ROOT"
$PY tools/experiments/prepare_sm_causal.py --out "$ROOT" --stage screen --steps 2340
sbatch --parsable scripts/slurm/14_sm_causal_gate.sh "$ROOT"
# After this GPU job passes, supply its returned job ID:
$PY tools/experiments/submit_sm_causal.py "$ROOT" --gate-job JOB_ID
```

The submitter freezes code, records hashes/jobs.json, and submits:
gate → four two-step DDP smoke runs → A0–A3 array → corresponding evaluations →
CPU paired report. Failed prerequisites cancel dependent jobs. Re-running the
submitter on an existing jobs.json refuses duplicate submission.

Outputs: `screen/runs/RUN/initialization.json`, training logs/checkpoints,
`evaluation/{best,last}/*_windows.csv`, and `screen/summary/report.md` with paired
spatial-block intervals. Check both Slurm exit status and completed markers.

Preparation also writes B25/B50/B100, C0/C1/C2/C2control, and Dfreeze configurations
for three seeds. **They are not automatically submitted**: the protocol is staged.
C submission fails until a real common-support eligibility audit is available;
the submitter schedules `audit_sm_temporal.py` on a CPU node to produce this
audit; do not manually fabricate `temporal_support_passed.json`. Exact native SM pixel
IDs are absent in stored metadata; B grouping uses conservative 5° block × day
groups and cannot establish native-observation independence. Full-budget configs
can be generated separately with `--stage full --steps 11700`; formal runs need
an explicit selected-run list/launcher, not the initial screen array launcher.

Not yet implemented: extra equal-count geographic-coverage contrasts, reduced
encoder-LR branch, expanded domain holdouts, and missing-exogenous-data tests.
These are follow-up branches, not claims made by the initial A-stage experiment.
