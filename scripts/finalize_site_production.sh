#!/bin/bash
# CPU-only multi-cohort/v3 publication after training statistics and mixed-data preflight.
set -euo pipefail
: "${HYDROSTATE_CODE_ROOT:?Set frozen code root}"
: "${HYDROSTATE_SOURCE_HASHES:?Set source checksum file}"
: "${HYDROSTATE_PYTHON:?Set CPU Python interpreter}"
: "${HYDROSTATE_CODE_COMMIT:?Set frozen code commit}"
cd "$HYDROSTATE_CODE_ROOT"
sha256sum --check --status "$HYDROSTATE_SOURCE_HASHES"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 BLOSC_NTHREADS=1 PYTHONUNBUFFERED=1
exec "$HYDROSTATE_PYTHON" tools/data/prepare_sites.py publish-joint \
    --code-commit "$HYDROSTATE_CODE_COMMIT" "$@"
