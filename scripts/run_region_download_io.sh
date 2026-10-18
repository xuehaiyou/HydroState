#!/bin/bash
# HPC2021 file-transfer nodes: bounded CPU use, shared storage, no Slurm allocation.
set -euo pipefail
case "$(hostname -s)" in
    hpc2021-io1|hpc2021-io2) ;;
    *) echo "Run regional downloads on an HPC2021 IO node." >&2; exit 64 ;;
esac
: "${HYDROSTATE_CODE_ROOT:?Set the frozen source directory}"
: "${HYDROSTATE_SOURCE_HASHES:?Set the source checksum file}"
: "${HYDROSTATE_GEE_PYTHON:?Set the Python interpreter with Earth Engine dependencies}"
cd "$HYDROSTATE_CODE_ROOT"
sha256sum --check --status "$HYDROSTATE_SOURCE_HASHES"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 BLOSC_NTHREADS=1 PYTHONUNBUFFERED=1
download_cpus="$("$HYDROSTATE_GEE_PYTHON" -c \
    'import os; print(",".join(str(cpu) for cpu in sorted(os.sched_getaffinity(0))[:2]))')"
exec nice -n 10 taskset -c "$download_cpus" "$HYDROSTATE_GEE_PYTHON" -u \
    tools/data/prepare_region_inputs.py --workers 4 --min-free-gib 20 "$@"
