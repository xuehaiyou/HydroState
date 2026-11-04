#!/bin/bash
# Bounded network collection; all heavy raw-data processing has already completed on CPU nodes.
set -euo pipefail
case "$(hostname -s)" in
    hpc2021-io1|hpc2021-io2) ;;
    *) echo "Run GEE station collection on an HPC2021 IO node." >&2; exit 64 ;;
esac
: "${HYDROSTATE_CODE_ROOT:?Set frozen code root}"
: "${HYDROSTATE_SOURCE_HASHES:?Set source checksum file}"
: "${HYDROSTATE_PYTHON:?Set Python interpreter}"
cd "$HYDROSTATE_CODE_ROOT"
sha256sum --check --status "$HYDROSTATE_SOURCE_HASHES"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 BLOSC_NTHREADS=1 PYTHONUNBUFFERED=1
site_cpus="$("$HYDROSTATE_PYTHON" -c \
    'import os; print(",".join(map(str, sorted(os.sched_getaffinity(0))[:2])))')"
exec nice -n 10 taskset -c "$site_cpus" "$HYDROSTATE_PYTHON" \
    tools/data/prepare_sites.py produce "$@"
