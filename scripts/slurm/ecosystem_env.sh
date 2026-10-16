#!/bin/bash
# Shared paths for statistics and training; source from the submitted scripts.
export HYDROSTATE_CODE_ROOT="${HYDROSTATE_CODE_ROOT:-/home/xiaoz/HydroState}"
cd "$HYDROSTATE_CODE_ROOT"
if [[ -n "${HYDROSTATE_CODE_COMMIT:-}" ]]; then
    [[ "$(git rev-parse HEAD)" == "$HYDROSTATE_CODE_COMMIT" ]] || {
        echo "Repository commit changed after submission; resubmit with the intended code." >&2
        exit 1
    }
    git diff --quiet HEAD -- || { echo "Tracked source has uncommitted changes." >&2; exit 1; }
fi
export HYDROSTATE_CONDA_BASE="${HYDROSTATE_CONDA_BASE:-/home/xiaoz/miniconda}"
export HYDROSTATE_CONDA_ENV=hydrostate
source "$HYDROSTATE_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$HYDROSTATE_CONDA_ENV"
export HYDROSTATE_DATA_ROOT="${HYDROSTATE_DATA_ROOT:-/scr/u/xiaoz/HydroState/samples_merged_v1}"
export HYDROSTATE_MANIFEST="${HYDROSTATE_MANIFEST:-$HYDROSTATE_DATA_ROOT/manifests/merged_v1_n20000/samples.parquet}"
export HYDROSTATE_TARGET_STATS="${HYDROSTATE_TARGET_STATS:-${HYDROSTATE_MANIFEST%/*}/target_stats_ecosystem_temporal_v1.json}"
export HYDROSTATE_LABEL_PACK="${HYDROSTATE_LABEL_PACK:-labels/ecosystem_v1}"
export HYDROSTATE_LST_PACK="${HYDROSTATE_LST_PACK:-labels/lst_observation_v1}"
export HYDROSTATE_MODEL_PATH="${HYDROSTATE_MODEL_PATH:-/scr/u/xiaoz/HydroState/pretrained/OlmoEarth-v1_2-Base}"
export HYDROSTATE_CONFIG=configs/hydrostate_ecosystem_temporal_hpc.py
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONUNBUFFERED=1
# MMEngine checkpoints include optimizer/config state, not only tensor weights.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
test -s "$HYDROSTATE_MANIFEST"
