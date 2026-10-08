#!/bin/bash
#SBATCH --job-name=hs_sm_gate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=l40s
#SBATCH --account=geog_geors
#SBATCH --qos=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --time=02:00:00
#SBATCH --mail-type=NONE
#SBATCH --output=/home/xiaoz/HydroState/logs/%x-%j.out
#SBATCH --error=/home/xiaoz/HydroState/logs/%x-%j.err
set -euo pipefail
root="${1:?experiment root}"
cd "${2:-/home/xiaoz/HydroState}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONUNBUFFERED=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
/home/xiaoz/miniconda/envs/hydrostate/bin/python tools/experiments/sm_gate.py \
  --snapshot /home/xiaoz/HydroState/outputs/diagnostics_v2_4107932/frozen_epoch32 \
  --out "$root/gate"
