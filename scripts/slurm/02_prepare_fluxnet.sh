#!/bin/bash
#SBATCH --job-name=hs_fluxnet
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
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

python tools/data/stations/prepare_fluxnet.py \
  --work-dir /fossfs/xiaozhen/HydroState/stations/fluxnet \
  --eligible-area /fossfs/xiaozhen/HydroState/catalog/coverage_v1/common_coverage.geojson \
  --download "$@"

python tools/data/stations/normalize_fluxnet_et.py \
  --input /fossfs/xiaozhen/HydroState/stations/fluxnet/downloads \
  --output /fossfs/xiaozhen/HydroState/stations/fluxnet/et_observations.parquet \
  "$@"
