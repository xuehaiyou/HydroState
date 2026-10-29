#!/usr/bin/env bash
#SBATCH --job-name=hs_region_profile
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=c_foss_gpu
#SBATCH --account=geog_geors
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --time=00:25:00
#SBATCH --mail-type=NONE
set -euo pipefail
: "${HYDROSTATE_CODE_ROOT:?Set an immutable source release}"
: "${HYDROSTATE_CODE_COMMIT:?Set the source commit}"
: "${HYDROSTATE_SOURCE_HASHES:?Set source checksums}"
if [[ "${SLURM_JOB_PARTITION:-}" != c_foss_gpu ]]; then
    echo 'FossFS profiling requires c_foss_gpu.' >&2
    exit 2
fi
cd "$HYDROSTATE_CODE_ROOT"
sha256sum --check --status "$HYDROSTATE_SOURCE_HASHES"
source "${HYDROSTATE_CONDA_BASE:-/home/xiaoz/miniconda}/etc/profile.d/conda.sh"
conda activate "${HYDROSTATE_CONDA_ENV:-hydrostate}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 BLOSC_NTHREADS=1 PYTHONUNBUFFERED=1
exec python tools/benchmarks/region_inference.py "$@"
