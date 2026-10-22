#!/bin/bash
#SBATCH --job-name=hs_sm_eval
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
#SBATCH --output=/home/xiaoz/HydroState/logs/%x-%A_%a.out
#SBATCH --error=/home/xiaoz/HydroState/logs/%x-%A_%a.err
set -euo pipefail
root="${1:?experiment root}"
cd "$root/code"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONUNBUFFERED=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
config="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$root/screen/initial_configs.txt")"
/home/xiaoz/miniconda/envs/hydrostate/bin/python tools/experiments/evaluate_sm_causal.py "$config"
