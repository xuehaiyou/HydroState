#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# Global production: seven equal frequency bins × year-month × climate strata.
# Native climate-grid screening skips ocean blocks before 30 m operations.
# Override defaults with environment variables if needed.
# Reuse the same output directory and sampling parameters to resume interrupted work.
GEE_PROJECT="${GEE_PROJECT:-cropland-477906}"
SAMPLES="${SAMPLES:-10000}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/d/Hydrostate/samples}"
CANDIDATE_DIR="${CANDIDATE_DIR:-${OUTPUT_DIR}/candidates}"
DRIVE_TOKEN="${DRIVE_TOKEN:-$PWD/drive_token.json}"
PYTHON="${PYTHON:-/home/xiaoz/miniconda3/envs/hydrostate/bin/python}"

args=(--project "$GEE_PROJECT"
      --output "$OUTPUT_DIR"
      --candidate-dir "$CANDIDATE_DIR" --candidate-seed "${CANDIDATE_SEED:-42}"
      --drive-token "$DRIVE_TOKEN"
      --start-year "${START_YEAR:-2019}" --end-year "${END_YEAR:-2024}"
      --samples "$SAMPLES" --seed "${SEED:-42}"
      --batch-size "${BATCH_SIZE:-32}" --concurrency "${CONCURRENCY:-2}"
      --candidates-per-class "${CANDIDATES_PER_CLASS:-16}"
      --poll-seconds "${POLL_SECONDS:-30}")

# Run in the foreground with unbuffered progress output.
exec "$PYTHON" -u hydrostate/data/gee_sampling.py "${args[@]}" "$@"
