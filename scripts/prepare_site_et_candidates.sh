#!/usr/bin/env bash
# Run in a CPU Slurm allocation. A bounded probe and the full run share checkpoints.
set -euo pipefail
if [[ $# -lt 4 || $# -gt 5 ]]; then
    echo "Usage: $0 CODE_ROOT SITE_ROOT OUTPUT AS_OF [PROBE_FILE_LIMIT]" >&2
    exit 2
fi
code_root=$1
site_root=$2
output=$3
as_of=$4
site_python=${HYDROSTATE_PYTHON:-/home/xiaoz/miniconda/envs/hydrostate/bin/python}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$code_root${PYTHONPATH:+:$PYTHONPATH}"
cd "$code_root"
probe_args=()
if [[ $# -eq 5 ]]; then probe_args=(--limit "$5"); fi
"$site_python" tools/data/prepare_sites.py et-candidates \
    --index "$site_root/index_v2_20261103" --splits "$site_root/splits_20261103" \
    --strata "$site_root/strata_20261103" --output "$output" --as-of "$as_of" \
    "${probe_args[@]}"
if [[ $# -eq 4 ]]; then
    "$site_python" tools/data/prepare_sites.py sample-capacity \
        --sm-candidates "$site_root/candidates_20261103/candidates.parquet" \
        --et-candidates "$output/candidates.parquet" \
        --footprint-candidates "$site_root/footprints_icos_20261103/candidates.parquet" \
        --output "$output/sampling" --seed 43 --gap-days 30
fi
