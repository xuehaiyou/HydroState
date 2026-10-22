#!/bin/bash
# CPU training statistics -> one 50-epoch GPU job, both on scratch data.
set -euo pipefail
export HYDROSTATE_CODE_ROOT="${HYDROSTATE_CODE_ROOT:-/home/xiaoz/HydroState}"
cd "$HYDROSTATE_CODE_ROOT"
git diff --quiet HEAD -- || { echo "Commit source changes before submitting." >&2; exit 1; }
export HYDROSTATE_CODE_COMMIT="$(git rev-parse HEAD)"
mkdir -p /scr/u/xiaoz/HydroState/logs /scr/u/xiaoz/HydroState/submissions
record="${HYDROSTATE_SUBMISSION_RECORD:-/scr/u/xiaoz/HydroState/submissions/ecosystem_temporal_${HYDROSTATE_CODE_COMMIT}.txt}"
# Preserve accepted job IDs on partial failure and refuse duplicate retries.
(set -o noclobber; printf 'CODE_COMMIT=%s\n' "$HYDROSTATE_CODE_COMMIT" > "$record")
export HYDROSTATE_SOURCE_HASHES="$record.source.sha256"
git ls-files -z | xargs -0 sha256sum > "$HYDROSTATE_SOURCE_HASHES"
stats=$(sbatch --parsable scripts/legacy/slurm/25_prepare_ecosystem_statistics.sh)
stats=${stats%%;*}
printf 'STATS_JOB=%s\n' "$stats" | tee -a "$record"
train=$(sbatch --parsable --dependency="afterok:$stats" --kill-on-invalid-dep=yes \
    scripts/legacy/slurm/26_train_ecosystem_temporal.sh)
train=${train%%;*}
printf 'TRAIN_JOB=%s\n' "$train" | tee -a "$record"
squeue -j "$stats,$train" -o '%.18i %.30j %.10T %.30R'
