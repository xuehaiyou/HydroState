#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON="${PYTHON:-/home/xiaoz/miniconda3/envs/hydrostate/bin/python}"
SAMPLE_ROOT="${SAMPLE_ROOT:-/mnt/d/Hydrostate/samples/merged_v1}"
OUTPUT_DIR="${OUTPUT_DIR:-$SAMPLE_ROOT/labels/ecosystem_v1}"
WORK_DIR="${WORK_DIR:-$SAMPLE_ROOT/labels/.ecosystem_clear_v1.work}"
exec "$PYTHON" -u tools/data/extend_gee_labels.py \
  --manifest "$SAMPLE_ROOT/samples.parquet" --output "$OUTPUT_DIR" \
  --work-dir "$WORK_DIR" \
  --project "${GEE_PROJECT:-cropland-477906}" --workers "${WORKERS:-8}" "$@"
