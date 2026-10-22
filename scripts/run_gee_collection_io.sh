#!/bin/bash
# Small-window GEE collection on network-enabled HPC2021 file-transfer nodes.
set -euo pipefail
case "$(hostname -s)" in
    hpc2021-io1|hpc2021-io2) ;;
    *) echo "Run sample collection on an HPC2021 IO node." >&2; exit 64 ;;
esac
: "${HYDROSTATE_CODE_ROOT:?Set the frozen source directory}"
: "${HYDROSTATE_SOURCE_HASHES:?Set the source checksum file}"
: "${HYDROSTATE_PYTHON:?Set the Python interpreter}"
: "${HYDROSTATE_DATA_ROOT:?Set the collection directory on shared scratch}"
: "${HYDROSTATE_CANDIDATE_ROOT:?Set the candidate cache on shared scratch}"
cd "$HYDROSTATE_CODE_ROOT"
sha256sum --check --status "$HYDROSTATE_SOURCE_HASHES"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 BLOSC_NTHREADS=1 PYTHONUNBUFFERED=1
collection_cpus="$("$HYDROSTATE_PYTHON" -c \
    'import os; print(",".join(str(cpu) for cpu in sorted(os.sched_getaffinity(0))[:2]))')"
exec nice -n 10 taskset -c "$collection_cpus" bash scripts/run_gee_sampling.sh "$@"
