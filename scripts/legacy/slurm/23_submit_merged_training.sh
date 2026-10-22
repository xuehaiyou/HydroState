#!/bin/bash
# Run on HPC: preparation -> GPU smoke check -> independent 50-epoch experiment.
set -euo pipefail
cd /home/xiaoz/HydroState
mkdir -p logs
prep=$(sbatch --parsable scripts/legacy/slurm/21_prepare_merged_samples.sh)
prep=${prep%%;*}
echo "PREP_JOB=$prep"
smoke=$(sbatch --parsable --dependency="afterok:$prep" --kill-on-invalid-dep=yes \
    --job-name=hs_merged_smoke --time=02:00:00 \
    --export=ALL,HYDROSTATE_CONFIG=configs/experiments/legacy/hydrostate_merged_smoke.py,HYDROSTATE_RUN_PREFIX=merged_smoke \
    scripts/legacy/slurm/22_train_merged_l40s.sh)
smoke=${smoke%%;*}
echo "SMOKE_JOB=$smoke"
train=$(sbatch --parsable --dependency="afterok:$smoke" --kill-on-invalid-dep=yes \
    --export=ALL,HYDROSTATE_CONFIG=configs/experiments/legacy/hydrostate_merged.py,HYDROSTATE_RUN_PREFIX=merged_n20000 \
    scripts/legacy/slurm/22_train_merged_l40s.sh)
train=${train%%;*}
echo "TRAIN_JOB=$train"
printf 'PREP_JOB=%s\nSMOKE_JOB=%s\nTRAIN_JOB=%s\n' "$prep" "$smoke" "$train" \
    > "logs/merged_submission_${train}.txt"
squeue -j "$prep,$smoke,$train" -o '%.18i %.22j %.10T %.30R'
