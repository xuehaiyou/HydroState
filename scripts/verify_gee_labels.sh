#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON="${PYTHON:-/home/xiaoz/miniconda3/envs/hydrostate/bin/python}"
SAMPLE_ROOT="${SAMPLE_ROOT:-/mnt/d/Hydrostate/samples/merged_v1}"
PILOT_DIR="${PILOT_DIR:-$SAMPLE_ROOT/labels/ecosystem_pilot_v2}"
WORK_DIR="${WORK_DIR:-$SAMPLE_ROOT/labels/.ecosystem_clear_v1.work}"

"$PYTHON" -m pytest -q
if [[ ! -f "$PILOT_DIR/_SUCCESS.json" ]]; then
  "$PYTHON" -u tools/data/extend_gee_labels.py \
    --manifest "$SAMPLE_ROOT/samples.parquet" --output "$PILOT_DIR" \
    --work-dir "$WORK_DIR" --project "${GEE_PROJECT:-cropland-477906}" \
    --workers "${WORKERS:-8}" --limit 64
fi
"$PYTHON" -u tools/data/validate_gee_labels.py \
  --manifest "$SAMPLE_ROOT/samples.parquet" --pack "$PILOT_DIR" --limit 0
