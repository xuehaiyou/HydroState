#!/usr/bin/env bash
# Collect a new batch, then validate/publish a combined manifest without rewriting Zarr.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHON="${PYTHON:-/home/xiaoz/miniconda3/envs/hydrostate/bin/python}"
export SAMPLES="${SAMPLES:-10000}"
export SEED="${SEED:-43}"
export SPLIT_SEED="${SPLIT_SEED:-42}"
export OUTPUT_DIR="${OUTPUT_DIR:-/mnt/d/Hydrostate/samples/batch_seed${SEED}_n${SAMPLES}}"
OLD_MANIFEST="${OLD_MANIFEST:-/mnt/d/Hydrostate/samples/samples.parquet}"
MERGED_DIR="${MERGED_DIR:-/mnt/d/Hydrostate/samples/merged_v1}"
# Refuse ambiguous restarts after publishing an immutable combined version.
if [[ -e "$MERGED_DIR" ]]; then
    echo "Combined version already exists: $MERGED_DIR; choose a new MERGED_DIR" >&2
    exit 1
fi
test -s "$OLD_MANIFEST"
bash scripts/run_gee_sampling.sh --exclude-manifest "$OLD_MANIFEST"
"$PYTHON" -u -m hydrostate.data.merge_samples \
    --source "$OLD_MANIFEST" --source "$OUTPUT_DIR/samples.parquet" \
    --split-seed "$SPLIT_SEED" --output "$MERGED_DIR" --copy-zarr
