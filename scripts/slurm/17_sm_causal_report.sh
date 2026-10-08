#!/bin/bash
#SBATCH --job-name=hs_sm_report
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=intel
#SBATCH --account=geog_geors
#SBATCH --qos=normal
#SBATCH --time=00:15:00
#SBATCH --mail-type=NONE
#SBATCH --output=/home/xiaoz/HydroState/logs/%x-%j.out
#SBATCH --error=/home/xiaoz/HydroState/logs/%x-%j.err
set -euo pipefail
root="${1:?experiment root}"
cd "$root/code"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
/home/xiaoz/miniconda/envs/hydrostate/bin/python tools/experiments/summarize_sm_causal.py "$root"
