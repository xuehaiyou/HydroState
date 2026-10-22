#!/bin/bash
#SBATCH --job-name=hs_poyang_predict
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --partition=l40s
#SBATCH --account=geog_geors
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=GPU_MEM:48GB
#SBATCH --time=04:00:00
#SBATCH --mail-type=NONE
#SBATCH --output=/scr/u/xiaoz/HydroState/logs/%x-%j.out
#SBATCH --error=/scr/u/xiaoz/HydroState/logs/%x-%j.err
set -euo pipefail
: "${HYDROSTATE_CODE_ROOT:?Submit with 29_submit_region.sh}"
: "${HYDROSTATE_SOURCE_HASHES:?Missing source provenance}"
cd "$HYDROSTATE_CODE_ROOT"
sha256sum --check --status "$HYDROSTATE_SOURCE_HASHES"
source /home/xiaoz/miniconda/etc/profile.d/conda.sh
conda activate hydrostate
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONUNBUFFERED=1
python tools/legacy/predict_region.py predict --workers 2 --batch-size 8 "$@"
