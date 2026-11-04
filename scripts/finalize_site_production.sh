#!/bin/bash
# CPU-only acceptance and SCR publication after collection and model pilot pass.
set -euo pipefail
: "${HYDROSTATE_CODE_ROOT:?Set frozen code root}"
: "${HYDROSTATE_SOURCE_HASHES:?Set source checksum file}"
: "${HYDROSTATE_PYTHON:?Set CPU Python interpreter}"
cd "$HYDROSTATE_CODE_ROOT"
sha256sum --check --status "$HYDROSTATE_SOURCE_HASHES"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 BLOSC_NTHREADS=1 PYTHONUNBUFFERED=1
exec "$HYDROSTATE_PYTHON" tools/data/prepare_sites.py publish "$@"
