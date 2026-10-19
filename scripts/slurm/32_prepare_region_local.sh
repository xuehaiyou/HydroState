#!/bin/bash
# Local raster decoding/reprojection needs no network access.
#SBATCH --job-name=hs_local_l2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --account=geog_geors
#SBATCH --partition=c_foss_amd
#SBATCH --qos=normal
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
: "${HYDROSTATE_CODE_ROOT:?Set a frozen source directory}"
: "${HYDROSTATE_SOURCE_HASHES:?Set source checksum file}"
PYTHON="${HYDROSTATE_GEE_PYTHON:-/scr/u/xiaoz/HydroState/environments/gee-download/bin/python}"
REGION_ROOT="${REGION_ROOT:-/scr/u/xiaoz/HydroState/inference/poyang_local_l2_201704_202512}"
cd "$HYDROSTATE_CODE_ROOT"
sha256sum --check --status "$HYDROSTATE_SOURCE_HASHES"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export GDAL_NUM_THREADS=1 BLOSC_NTHREADS=1 PYTHONUNBUFFERED=1
export GDAL_CACHEMAX=128
exec "$PYTHON" -u tools/data/prepare_region_inputs.py \
  --root "$REGION_ROOT" --backend local \
  --local-config configs/data/hpc_local_l2.json \
  --workers "${SLURM_CPUS_PER_TASK:-4}" --min-free-gib 20 "$@"
