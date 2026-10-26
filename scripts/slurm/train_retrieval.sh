#!/bin/bash
# Submit only after collection audit and training-only statistics have completed.
# Pass mtlora, lora or full; keep the data/statistics fixed across controls.
#SBATCH --job-name=hs_mtlora
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --partition=l40s
#SBATCH --account=geog_geors
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=GPU_MEM:48GB
#SBATCH --time=96:00:00
#SBATCH --mail-type=NONE
set -euo pipefail

case "${1:-}" in
    mtlora) export HYDROSTATE_CONFIG=configs/hydrostate_mtlora.py ;;
    lora) export HYDROSTATE_CONFIG=configs/hydrostate_mtlora_shared.py ;;
    full) export HYDROSTATE_CONFIG=configs/hydrostate_full.py ;;
    *) echo 'Expected architecture: mtlora, lora or full' >&2; exit 2 ;;
esac
shift
: "${HYDROSTATE_CODE_ROOT:?Set an immutable source release}"
: "${HYDROSTATE_CODE_COMMIT:?Set the release commit}"
: "${HYDROSTATE_SOURCE_HASHES:?Set release source checksums}"
: "${HYDROSTATE_DATA_ROOT:?Set the completed collection root}"
: "${HYDROSTATE_TARGET_STATS:?Prepare temporal training statistics first}"
: "${HYDROSTATE_WORK_DIR:?Set a separate output directory per architecture}"
if [[ "${SLURM_CPUS_PER_TASK:-6}" -gt 6 ]]; then
    echo 'This single-GPU experiment must request at most 6 CPU cores.' >&2
    exit 2
fi
export HYDROSTATE_MANIFEST="$HYDROSTATE_DATA_ROOT/samples.parquet"
export HYDROSTATE_MODEL_PATH="${HYDROSTATE_MODEL_PATH:-/scr/u/xiaoz/HydroState/pretrained/OlmoEarth-v1_2-Base}"
cd "$HYDROSTATE_CODE_ROOT"
sha256sum --check --status "$HYDROSTATE_SOURCE_HASHES"
source "${HYDROSTATE_CONDA_BASE:-/home/xiaoz/miniconda}/etc/profile.d/conda.sh"
conda activate "${HYDROSTATE_CONDA_ENV:-hydrostate}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export BLOSC_NTHREADS=1 PYTHONUNBUFFERED=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export HYDROSTATE_GPUS_PER_NODE=1 HYDROSTATE_PRECISION=bfloat16

# Refuse a live/partial collection or statistics from a different manifest.
# The training CLI additionally validates the full statistics data contract.
python tools/data/prepare_training.py verify \
    --data-root "$HYDROSTATE_DATA_ROOT" --statistics "$HYDROSTATE_TARGET_STATS"

mkdir -p "$HYDROSTATE_WORK_DIR/provenance"
printf '%s\n' "$HYDROSTATE_CODE_COMMIT" > "$HYDROSTATE_WORK_DIR/provenance/code_commit.txt"
cp "$HYDROSTATE_DATA_ROOT/_COLLECTION_SUCCESS.json" "$HYDROSTATE_TARGET_STATS" \
    "$HYDROSTATE_WORK_DIR/provenance/"
exec bash scripts/train.sh \
    train_dataloader.num_workers=4 val_dataloader.num_workers=4 test_dataloader.num_workers=4 "$@"
