#!/bin/bash
#SBATCH --job-name=hs_l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=l40s
#SBATCH --account=geog_geors
#SBATCH --qos=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --time=24:00:00
#SBATCH --mail-type=NONE
#SBATCH --output=/home/xiaoz/HydroState/logs/%x-%j.out
#SBATCH --error=/home/xiaoz/HydroState/logs/%x-%j.err
set -euo pipefail
cd "${HYDROSTATE_CODE_ROOT:-/home/xiaoz/HydroState}"
export HYDROSTATE_CONDA_BASE="${HYDROSTATE_CONDA_BASE:-/home/xiaoz/miniconda}"
export HYDROSTATE_CONDA_ENV=hydrostate
export HYDROSTATE_DATA_ROOT="${HYDROSTATE_DATA_ROOT:-/scr/u/xiaoz/HydroState/samples_v1}"
export HYDROSTATE_MANIFEST="${HYDROSTATE_MANIFEST:-$HYDROSTATE_DATA_ROOT/manifests/gee_20260922_n1024/samples.parquet}"
export HYDROSTATE_MODEL_PATH="${HYDROSTATE_MODEL_PATH:-/scr/u/xiaoz/HydroState/pretrained/OlmoEarth-v1_2-Base}"
export HYDROSTATE_TARGET_STATS="${HYDROSTATE_TARGET_STATS:-${HYDROSTATE_MANIFEST%/*}/target_stats.json}"
export HYDROSTATE_CONFIG="${HYDROSTATE_CONFIG:-configs/hydrostate_l40s.py}"
export HYDROSTATE_WORK_DIR="${HYDROSTATE_WORK_DIR:-/scr/u/xiaoz/HydroState/work_dirs/nll_${SLURM_JOB_ID}}"
# Fixed validation samples per epoch; 0 disables WF/SM/ET comparison PNGs.
# Columns: 30 m prediction, supervision-scale prediction, ground truth.
export HYDROSTATE_VIS_NUM_SAMPLES="${HYDROSTATE_VIS_NUM_SAMPLES:-4}"
# GPU count is detected by the shared launcher; an explicit env value is optional.
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
test -s "$HYDROSTATE_TARGET_STATS"
bash scripts/slurm/11_train_hpc.sh "$@"
