#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PYTHON="${PYTHON:-/home/xiaoz/miniconda3/envs/hydrostate/bin/python}"
SAMPLE_ROOT="${SAMPLE_ROOT:-/mnt/d/Hydrostate/samples/merged_v1}"
OUTPUT_DIR="${OUTPUT_DIR:-$SAMPLE_ROOT/labels/lst_observation_v1}"
exec "$PYTHON" -u tools/legacy/data/extend_gee_lst.py \
  --manifest "$SAMPLE_ROOT/samples.parquet" \
  --base-pack "${BASE_PACK:-$SAMPLE_ROOT/labels/ecosystem_v1}" \
  --output "$OUTPUT_DIR" \
  --work-dir "${WORK_DIR:-$SAMPLE_ROOT/labels/.lst_observation_v1.work}" \
  --project "${GEE_PROJECT:-cropland-477906}" --workers "${WORKERS:-8}" "$@"
