#!/bin/bash
#SBATCH --job-name=hs_ecosystem_temporal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=l40s,c_foss_gpu,condo_gpu
#SBATCH --account=geog_geors
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=GPU_MEM:48GB
#SBATCH --time=96:00:00
#SBATCH --mail-type=NONE
#SBATCH --output=/scr/u/xiaoz/HydroState/logs/%x-%j.out
#SBATCH --error=/scr/u/xiaoz/HydroState/logs/%x-%j.err
set -euo pipefail
source "${HYDROSTATE_CODE_ROOT:-/home/xiaoz/HydroState}/scripts/slurm/ecosystem_env.sh"
export HYDROSTATE_WORK_DIR="${HYDROSTATE_WORK_DIR:-/scr/u/xiaoz/HydroState/work_dirs/ecosystem_temporal_${SLURM_JOB_ID}}"
export HYDROSTATE_PRECISION=bfloat16
export HYDROSTATE_GPUS_PER_NODE=1
test -s "$HYDROSTATE_TARGET_STATS"
mkdir -p "$HYDROSTATE_WORK_DIR/provenance"
git rev-parse HEAD > "$HYDROSTATE_WORK_DIR/provenance/code_commit.txt"
cp "$HYDROSTATE_TARGET_STATS" "$HYDROSTATE_WORK_DIR/provenance/target_statistics.json"
sha256sum "$HYDROSTATE_MANIFEST" > "$HYDROSTATE_WORK_DIR/provenance/manifest.sha256"
exec bash scripts/slurm/11_train_hpc.sh "$@"
