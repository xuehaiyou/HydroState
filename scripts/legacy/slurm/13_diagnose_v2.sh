#!/bin/bash
#SBATCH --job-name=hs_v2_diagnostics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=l40s
#SBATCH --account=geog_geors
#SBATCH --qos=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --time=08:00:00
#SBATCH --mail-type=NONE
#SBATCH --output=/home/xiaoz/HydroState/logs/%x-%j.out
#SBATCH --error=/home/xiaoz/HydroState/logs/%x-%j.err
set -euo pipefail
cd /home/xiaoz/HydroState
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONUNBUFFERED=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
snapshot="${1:?Provide frozen diagnostic snapshot directory}"
report="$snapshot/job_${SLURM_JOB_ID}"
mkdir -p "$report"
/home/xiaoz/miniconda/envs/hydrostate/bin/python tools/validation/diagnose_multitask.py \
    --config "$snapshot/hydrostate_l40s.py" \
    --checkpoints "$snapshot/best.pth" "$snapshot/latest.pth" \
    --out "$report" --gradient-batches-per-seed "${GRADIENT_BATCHES_PER_SEED:-34}" \
    --max-windows "${MAX_WINDOWS:-0}" --consistency-windows "${CONSISTENCY_WINDOWS:-64}"
/home/xiaoz/miniconda/envs/hydrostate/bin/python tools/validation/summarize_multitask_diagnostics.py "$report"
