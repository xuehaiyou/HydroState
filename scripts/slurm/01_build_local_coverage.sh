#!/bin/bash
#SBATCH --job-name=hs_coverage
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --account=geog_geors
#SBATCH --partition=c_foss_amd
#SBATCH --qos=normal
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
source /home/xiaoz/miniconda/etc/profile.d/conda.sh
conda activate hydrostate
cd /home/xiaoz/HydroState
mkdir -p /fossfs/xiaozhen/HydroState/logs

export OMP_NUM_THREADS=1
export GDAL_NUM_THREADS=1
python tools/data/build_local_coverage.py \
  --s1 /fossfs/QianWang/Download/Sentinel1/tile \
  --s2 /fossfs/DATAPOOL/Sentinel2_L2 \
  --landsat /fossfs/DATAPOOL/Landsat_L2 \
  --output /fossfs/xiaozhen/HydroState/catalog/coverage_v1 \
  --workers "${SLURM_CPUS_PER_TASK:-32}" "$@"
