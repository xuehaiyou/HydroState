#!/usr/bin/env bash
# Run inside a Slurm allocation with persistent FossFS storage mounted.
set -euo pipefail
if [[ $# -ne 3 ]]; then
    echo "Usage: $0 CODE_ROOT RAW_ROOT REPORT_PATH" >&2
    exit 2
fi
code_root=$1
raw_root=$2
report_path=$3
site_python=${HYDROSTATE_PYTHON:-/home/xiaoz/miniconda/envs/hydrostate/bin/python}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$code_root${PYTHONPATH:+:$PYTHONPATH}"
cd "$code_root"
mkdir -p "$(dirname "$report_path")"
"$site_python" tools/data/prepare_sites.py inventory --root "$raw_root" --output "${report_path%.json}_inventory.json"
"$site_python" tools/data/prepare_sites.py extract --root "$raw_root" --output "$report_path"
