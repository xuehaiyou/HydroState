#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PYTHON="${PYTHON:-/home/xiaoz/miniconda3/envs/hydrostate/bin/python}"
SAMPLE_ROOT="${SAMPLE_ROOT:-/mnt/d/Hydrostate/samples/merged_v1}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON" -u tools/legacy/data/extend_gee_landsat_l2.py \
  --manifest "${MANIFEST:-$SAMPLE_ROOT/samples.parquet}" \
  --output "${OUTPUT_DIR:-$SAMPLE_ROOT/inputs/landsat_l2_sr_v1}" \
  --work-dir "${WORK_DIR:-$SAMPLE_ROOT/inputs/.landsat_l2_sr_v1.work}" \
  --project "${GEE_PROJECT:-cropland-477906}" --workers "${WORKERS:-4}" "$@"
