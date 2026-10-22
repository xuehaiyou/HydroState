#!/bin/bash
#SBATCH --job-name=hs_target_stats
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=c_foss_amd
#SBATCH --account=geog_geors
#SBATCH --qos=debug
#SBATCH --time=00:30:00
#SBATCH --mail-type=NONE
#SBATCH --output=/home/xiaoz/HydroState/logs/%x-%j.out
#SBATCH --error=/home/xiaoz/HydroState/logs/%x-%j.err
set -euo pipefail
cd /home/xiaoz/HydroState
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
data_root="${HYDROSTATE_DATA_ROOT:-/scr/u/xiaoz/HydroState/samples_v1}"
manifest="${HYDROSTATE_MANIFEST:-$data_root/manifests/gee_20260922_n1024/samples.parquet}"
exec /home/xiaoz/miniconda/envs/hydrostate/bin/python -u tools/legacy/data/compute_target_statistics.py \
    --manifest "$manifest" --data-root "$data_root" --num-workers=4 \
    --output "${HYDROSTATE_TARGET_STATS:-${manifest%/*}/target_stats.json}"
