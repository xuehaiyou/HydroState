#!/usr/bin/env bash
# Collect a new batch, then validate/publish a combined manifest without rewriting Zarr.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PYTHON="${PYTHON:-/home/xiaoz/miniconda3/envs/hydrostate/bin/python}"
export SAMPLES="${SAMPLES:-10000}"
export SEED="${SEED:-43}"
export SPLIT_SEED="${SPLIT_SEED:-42}"
# This script expands the historical library. Centered sampling is a new
# independent version, not an append to these old inputs.
export SAMPLING_CONFIG="${SAMPLING_CONFIG:-configs/experiments/legacy/data/sampling/past16d.json}"
export OUTPUT_DIR="${OUTPUT_DIR:-/mnt/d/Hydrostate/samples/batch_seed${SEED}_n${SAMPLES}}"
OLD_MANIFEST="${OLD_MANIFEST:-/mnt/d/Hydrostate/samples/samples.parquet}"
MERGED_DIR="${MERGED_DIR:-/mnt/d/Hydrostate/samples/merged_v1}"
# Refuse ambiguous restarts after publishing an immutable combined version.
if [[ -e "$MERGED_DIR" ]]; then
    echo "Combined version already exists: $MERGED_DIR; choose a new MERGED_DIR" >&2
    exit 1
fi
test -s "$OLD_MANIFEST"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON" -u tools/legacy/data/prepare_gee_samples.py \
    --project "${GEE_PROJECT:-cropland-477906}" --output "$OUTPUT_DIR" \
    --samples "$SAMPLES" --seed "$SEED" --split-seed "$SPLIT_SEED" \
    --sampling-config "$SAMPLING_CONFIG" --exclude-manifest "$OLD_MANIFEST"
"$PYTHON" -u -m hydrostate.data.merge_samples \
    --source "$OLD_MANIFEST" --source "$OUTPUT_DIR/samples.parquet" \
    --split-seed "$SPLIT_SEED" --output "$MERGED_DIR" --copy-zarr
