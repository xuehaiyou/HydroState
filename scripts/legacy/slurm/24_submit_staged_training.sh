#!/bin/bash
# Reuse the prepared n=20000 dataset and existing launcher: smoke -> 50 epochs.
# Point HYDROSTATE_CODE_ROOT at a frozen source snapshot before submitting.
set -euo pipefail
export HYDROSTATE_CODE_ROOT="${HYDROSTATE_CODE_ROOT:-/home/xiaoz/HydroState}"
cd "$HYDROSTATE_CODE_ROOT"
record="${HYDROSTATE_SUBMISSION_RECORD:-$PWD/staged_submission.txt}"
# An exclusive record prevents accidental duplicate submission on retries.
(set -o noclobber; printf 'CODE_ROOT=%s\n' "$PWD" > "$record")
smoke=$(sbatch --parsable --job-name=hs_staged_sigmoid_smoke --time=02:00:00 \
    --export=ALL,HYDROSTATE_CONFIG=configs/experiments/legacy/hydrostate_staged_smoke.py,HYDROSTATE_RUN_PREFIX=staged_sigmoid_smoke \
    scripts/legacy/slurm/22_train_merged_l40s.sh)
smoke=${smoke%%;*}
printf 'SMOKE_JOB=%s\n' "$smoke" | tee -a "$record"
train=$(sbatch --parsable --dependency="afterok:$smoke" --kill-on-invalid-dep=yes \
    --job-name=hs_staged_sigmoid_n20000 \
    --export=ALL,HYDROSTATE_CONFIG=configs/experiments/legacy/hydrostate_staged.py,HYDROSTATE_RUN_PREFIX=staged_sigmoid_n20000 \
    scripts/legacy/slurm/22_train_merged_l40s.sh)
train=${train%%;*}
printf 'TRAIN_JOB=%s\n' "$train" | tee -a "$record"
squeue -j "$smoke,$train" -o '%.18i %.30j %.10T %.30R'
