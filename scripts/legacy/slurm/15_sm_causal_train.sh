#!/bin/bash
#SBATCH --job-name=hs_sm_causal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --partition=l40s
#SBATCH --account=geog_geors
#SBATCH --qos=gpu
#SBATCH --gres=gpu:nvidia_l40s:2
#SBATCH --time=48:00:00
#SBATCH --mail-type=NONE
#SBATCH --output=/home/xiaoz/HydroState/logs/%x-%A_%a.out
#SBATCH --error=/home/xiaoz/HydroState/logs/%x-%A_%a.err
set -euo pipefail
root="${1:?experiment root}"
mode="${2:-train}"
cd "$root/code"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONUNBUFFERED=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
python_bin=/home/xiaoz/miniconda/envs/hydrostate/bin/python
if [[ "$mode" == smoke ]]; then
  while IFS= read -r config; do
    "$python_bin" -m torch.distributed.run --standalone --nproc_per_node=2 \
      tools/experiments/train_sm_causal.py "$config" --smoke
  done < "$root/screen/initial_configs.txt"
  "$python_bin" tools/experiments/verify_sm_smoke.py "$root"
else
  config="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$root/screen/initial_configs.txt")"
  test -n "$config"
  "$python_bin" -m torch.distributed.run --standalone --nproc_per_node=2 \
    tools/experiments/train_sm_causal.py "$config"
fi
