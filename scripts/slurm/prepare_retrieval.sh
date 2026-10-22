#!/bin/bash
#SBATCH --job-name=hs_mtlora_prepare
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --partition=intel
#SBATCH --account=geog_geors
#SBATCH --qos=normal
#SBATCH --time=24:00:00
#SBATCH --mail-type=NONE
set -euo pipefail
cd "${HYDROSTATE_CODE_ROOT:?}"
sha256sum --check --status "${HYDROSTATE_SOURCE_HASHES:?}"
source "${HYDROSTATE_CONDA_BASE:-/home/xiaoz/miniconda}/etc/profile.d/conda.sh"
conda activate "${HYDROSTATE_CONDA_ENV:-hydrostate}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 BLOSC_NTHREADS=1
python tools/data/prepare_training.py prepare \
    --data-root "${HYDROSTATE_DATA_ROOT:?}" \
    --statistics "${HYDROSTATE_TARGET_STATS:?}" \
    --run-root "${HYDROSTATE_RUN_ROOT:?}" --num-workers 4
