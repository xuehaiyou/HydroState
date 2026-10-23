#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# Current contract: eight centre-source quotas, five 8-day mean composites.
# Change collection parameters in the config; a resumed directory is immutable.
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
HYDROSTATE_PYTHON="${HYDROSTATE_PYTHON:-python}"
COLLECTION_CONFIG="${COLLECTION_CONFIG:-configs/data/collection/retrieval.py}"
args=(--config "$COLLECTION_CONFIG")
if [[ -n "${HYDROSTATE_DATA_ROOT:-}" ]]; then args+=(--output "$HYDROSTATE_DATA_ROOT"); fi
if [[ -n "${HYDROSTATE_CANDIDATE_ROOT:-}" ]]; then args+=(--candidate-dir "$HYDROSTATE_CANDIDATE_ROOT"); fi
if [[ -n "${SAMPLES:-}" ]]; then args+=(--samples "$SAMPLES"); fi
if [[ -n "${GEE_PROJECT:-}" ]]; then args+=(--project "$GEE_PROJECT"); fi
exec "$HYDROSTATE_PYTHON" -u tools/data/collect_gee_dataset.py "${args[@]}" "$@"
