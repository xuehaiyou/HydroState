#!/bin/bash
#SBATCH --job-name=hs_poyang_inputs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=intel
#SBATCH --account=geog_geors
#SBATCH --qos=normal
#SBATCH --time=7-00:00:00
#SBATCH --mail-type=NONE
#SBATCH --output=/scr/u/xiaoz/HydroState/logs/%x-%j.out
#SBATCH --error=/scr/u/xiaoz/HydroState/logs/%x-%j.err
set -euo pipefail
: "${HYDROSTATE_CODE_ROOT:?Set the frozen source directory}"
: "${HYDROSTATE_SOURCE_HASHES:?Set the source checksum file}"
: "${HYDROSTATE_GEE_PYTHON:?Set the Python interpreter with Earth Engine and project dependencies}"
cd "$HYDROSTATE_CODE_ROOT"
sha256sum --check --status "$HYDROSTATE_SOURCE_HASHES"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONUNBUFFERED=1
exec "$HYDROSTATE_GEE_PYTHON" tools/legacy/predict_region.py prepare \
    --workers "${SLURM_CPUS_PER_TASK:-4}" --min-free-gib 20 "$@"
