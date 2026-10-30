#!/usr/bin/env bash
#SBATCH --job-name=hydro-poyang-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=32G
#SBATCH --partition=c_foss_gpu
#SBATCH --account=geog_geors
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mail-type=NONE
set -euo pipefail

: "${HYDROSTATE_CODE_ROOT:?Set an immutable source release}"
: "${HYDROSTATE_CODE_COMMIT:?Set the release commit}"
: "${HYDROSTATE_SOURCE_HASHES:?Set release source checksums}"
: "${HYDROSTATE_REGION_ROOT:?Set the prepared regional input root}"
: "${HYDROSTATE_REGION_MODEL_CONFIG:?Set the trained model config}"
: "${HYDROSTATE_CHECKPOINT:?Set the selected trained checkpoint}"
: "${HYDROSTATE_MODEL_PATH:?Set the pretrained backbone directory}"
: "${HYDROSTATE_TARGET_STATS:?Set the frozen training statistics}"
: "${HYDROSTATE_RUN_DIR:?Set a unique inference run directory}"
: "${HYDROSTATE_BATCH_SIZE:?Set the batch validated in a debug allocation}"
: "${HYDROSTATE_WORKERS:?Set the measured reader worker count}"
: "${HYDROSTATE_ESTIMATED_SECONDS:?Set end-to-end work estimate, excluding the 10 percent margin}"
if [[ "${SLURM_JOB_PARTITION:-}" != c_foss_gpu || "${SLURM_CPUS_PER_TASK:-3}" -gt 6 ]]; then
    echo 'FossFS inference requires c_foss_gpu and at most 6 CPU cores per GPU.' >&2
    exit 2
fi
test -r "$HYDROSTATE_REGION_ROOT/plan.json"
test -r "$HYDROSTATE_CHECKPOINT"
test -d "$HYDROSTATE_MODEL_PATH"
mkdir -p "$HYDROSTATE_RUN_DIR"
cd "$HYDROSTATE_CODE_ROOT"
sha256sum --check --status "$HYDROSTATE_SOURCE_HASHES"
source "${HYDROSTATE_CONDA_BASE:-/home/xiaoz/miniconda}/etc/profile.d/conda.sh"
conda activate "${HYDROSTATE_CONDA_ENV:-hydrostate}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 BLOSC_NTHREADS=1 PYTHONUNBUFFERED=1
exec srun --gpu-bind=single:1 python tools/predict_region.py \
    --root "$HYDROSTATE_REGION_ROOT" \
    --model-config "$HYDROSTATE_REGION_MODEL_CONFIG" \
    --checkpoint "$HYDROSTATE_CHECKPOINT" \
    --pretrained "$HYDROSTATE_MODEL_PATH" \
    --statistics "$HYDROSTATE_TARGET_STATS" \
    --run-dir "$HYDROSTATE_RUN_DIR" \
    --device cuda --workers "$HYDROSTATE_WORKERS" --batch-size "$HYDROSTATE_BATCH_SIZE" "$@"
