#!/bin/bash
#SBATCH --job-name=hs_nll_tests
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
exec /home/xiaoz/miniconda/envs/hydrostate/bin/python -m pytest -q tests \
    --junitxml="logs/nll-tests-${SLURM_JOB_ID}.xml"
