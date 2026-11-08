#!/bin/bash
# Run inside a single-GPU c_foss_gpu allocation after a representative GPU batch probe.
set -euo pipefail
: "${HYDROSTATE_CODE_ROOT:?Set frozen code release}"
: "${HYDROSTATE_CODE_COMMIT:?Set release commit}"
: "${HYDROSTATE_SOURCE_HASHES:?Set source checksum file}"
: "${HYDROSTATE_DATA_ROOT:?Set original product collection root}"
: "${HYDROSTATE_TARGET_STATS:?Set original product training statistics}"
: "${HYDROSTATE_SITE_SPLIT_ROOT:?Set completed v3 station split directory}"
: "${HYDROSTATE_WORK_DIR:?Set new joint experiment directory}"
: "${HYDROSTATE_MICRO_BATCH:?Set GPU-validated batch size}"
: "${HYDROSTATE_ACCUMULATIVE_COUNTS:?Set gradient accumulation to preserve effective batch}"
if [[ "${SLURM_JOB_PARTITION:-}" != c_foss_gpu ]]; then
    echo 'FossFS joint training requires a c_foss_gpu allocation.' >&2
    exit 2
fi
if [[ "${SLURM_CPUS_PER_TASK:-1}" -gt 6 ]]; then
    echo 'Single GPU jobs must use at most 6 CPUs.' >&2
    exit 2
fi
cd "$HYDROSTATE_CODE_ROOT"
sha256sum --check --status "$HYDROSTATE_SOURCE_HASHES"
source "${HYDROSTATE_CONDA_BASE:-/home/xiaoz/miniconda}/etc/profile.d/conda.sh"
conda activate "${HYDROSTATE_CONDA_ENV:-hydrostate}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export HYDROSTATE_CONFIG=configs/hydrostate_joint.py HYDROSTATE_GPUS_PER_NODE=1
export HYDROSTATE_MANIFEST="${HYDROSTATE_MANIFEST:-$HYDROSTATE_DATA_ROOT/samples.parquet}"
mkdir -p "$HYDROSTATE_WORK_DIR/provenance"
python tools/data/verify_joint_training.py --config "$HYDROSTATE_CONFIG" \
    --output "$HYDROSTATE_WORK_DIR/provenance/joint_preflight.json" --check-batch
printf '%s\n' "$HYDROSTATE_CODE_COMMIT" > "$HYDROSTATE_WORK_DIR/provenance/code_commit.txt"
cp "$HYDROSTATE_SITE_SPLIT_ROOT/datasets.json" "$HYDROSTATE_SITE_SPLIT_ROOT/split_report.json" \
   "$HYDROSTATE_SITE_SPLIT_ROOT/site_target_stats.json" "$HYDROSTATE_WORK_DIR/provenance/"
exec bash scripts/train.sh \
    "train_dataloader.batch_size=$HYDROSTATE_MICRO_BATCH" \
    "optim_wrapper.accumulative_counts=$HYDROSTATE_ACCUMULATIVE_COUNTS" \
    train_dataloader.num_workers=2 val_dataloader.num_workers=2 test_dataloader.num_workers=2 "$@"
