#!/bin/bash
#SBATCH --job-name=hs_ismn
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --account=geog_geors
#SBATCH --partition=c_foss_amd
#SBATCH --qos=normal
#SBATCH --time=24:00:00
#SBATCH --output=/fossfs/xiaozhen/HydroState/logs/%x-%j.out

set -euo pipefail
source /home/xiaoz/miniconda/etc/profile.d/conda.sh
conda activate hydrostate
cd /home/xiaoz/HydroState
mkdir -p /fossfs/xiaozhen/HydroState/logs

: "${ISMN_ARCHIVE:?Set ISMN_ARCHIVE to the downloaded ISMN zip or directory}"
python tools/data/stations/import_ismn.py \
  --archive "${ISMN_ARCHIVE}" \
  --eligible-area /fossfs/xiaozhen/HydroState/catalog/coverage_v1/common_coverage.geojson \
  --output /fossfs/xiaozhen/HydroState/stations/ismn \
  --min-depth 0 --max-depth 0.10 "$@"
