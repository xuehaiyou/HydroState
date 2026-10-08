#!/bin/bash
#SBATCH --job-name=hs_merged_n20000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --partition=l40s
#SBATCH --account=geog_geors
#SBATCH --qos=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --time=96:00:00
#SBATCH --mail-type=NONE
#SBATCH --output=/home/xiaoz/HydroState/logs/%x-%j.out
#SBATCH --error=/home/xiaoz/HydroState/logs/%x-%j.err
set -euo pipefail
cd /home/xiaoz/HydroState
export HYDROSTATE_DATA_ROOT=/scr/u/xiaoz/HydroState/samples_merged_v1
export HYDROSTATE_MANIFEST="$HYDROSTATE_DATA_ROOT/manifests/merged_v1_n20000/samples.parquet"
export HYDROSTATE_TARGET_STATS="${HYDROSTATE_MANIFEST%/*}/target_stats.json"
export HYDROSTATE_CONFIG="${HYDROSTATE_CONFIG:-configs/hydrostate_merged.py}"
export HYDROSTATE_WORK_DIR="/scr/u/xiaoz/HydroState/work_dirs/${HYDROSTATE_RUN_PREFIX:-merged_n20000}_${SLURM_JOB_ID}"
export HYDROSTATE_MODEL_PATH=/scr/u/xiaoz/HydroState/pretrained/OlmoEarth-v1_2-Base
export HYDROSTATE_PRECISION=bfloat16
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
test -s "${HYDROSTATE_MANIFEST%/*}/training_ready.json"
mkdir -p "$HYDROSTATE_WORK_DIR"
cp "${HYDROSTATE_MANIFEST%/*}/training_ready.json" "$HYDROSTATE_WORK_DIR/dataset_provenance.json"
# Reuse the established GPU checks, environment and single-GPU/DDP launcher.
exec bash scripts/slurm/12_train_l40s.sh "$@"
