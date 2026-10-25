#!/bin/bash
# Network-enabled IO preparation; CPU only, bounded work inside the Python queue.
set -euo pipefail
: "${HYDROSTATE_CODE_ROOT:?Set the committed code directory}"
: "${HYDROSTATE_PYTHON:?Set the Python interpreter}"
cd "$HYDROSTATE_CODE_ROOT"
if [[ -n "${HYDROSTATE_SOURCE_HASHES:-}" ]]; then
    sha256sum --check --status "$HYDROSTATE_SOURCE_HASHES"
fi
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 BLOSC_NTHREADS=1 PYTHONUNBUFFERED=1
region_cpus="$("$HYDROSTATE_PYTHON" -c \
    'import os; print(",".join(str(cpu) for cpu in sorted(os.sched_getaffinity(0))[:2]))')"
exec nice -n 10 taskset -c "$region_cpus" "$HYDROSTATE_PYTHON" \
    tools/prepare_region.py prepare "$@"
