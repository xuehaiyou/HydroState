#!/usr/bin/env bash
# CPU Slurm allocation; raw archives remain permanently extracted on FossFS.
set -euo pipefail
if [[ $# -lt 4 ]]; then
    echo "Usage: $0 CODE_ROOT SITE_ROOT OUTPUT AS_OF [icos-footprints options]" >&2
    exit 2
fi
code_root=$1
site_root=$2
output=$3
as_of=$4
shift 4
site_python=${HYDROSTATE_PYTHON:-/home/xiaoz/miniconda/envs/hydrostate/bin/python}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$code_root${PYTHONPATH:+:$PYTHONPATH}"
cd "$code_root"
"$site_python" tools/data/prepare_sites.py icos-footprints \
    --index "$site_root/index_v2_20261103" --splits "$site_root/splits_20261103" \
    --strata "$site_root/strata_20261103" --output "$output" --as-of "$as_of" "$@"
"$site_python" tools/data/prepare_sites.py availability \
    --sm-candidates "$site_root/candidates_20261103/candidates.parquet" \
    --et-candidates "$output/candidates.parquet" --output "$output/availability"
