#!/bin/bash
# Usage: bash scripts/slurm/29_submit_region.sh ROOT [--dates DATE ...] [--drop-s2]
# Prepared input caches are required. No GEE credentials are needed on the GPU.
set -euo pipefail
root="${1:?Provide the prepared regional root}"
shift
root="$(realpath "$root")"
cd "${HYDROSTATE_CODE_ROOT:-/home/xiaoz/HydroState}"
git diff --quiet HEAD -- || { echo "Commit code changes before submitting." >&2; exit 1; }
commit="$(git rev-parse HEAD)"
snapshot="/home/xiaoz/HydroState/submissions/region_${commit}"
mkdir -p /scr/u/xiaoz/HydroState/logs "$root/submissions"
if [ ! -d "$snapshot" ]; then
    temporary="${snapshot}.partial.$$"
    mkdir -p "$temporary"
    git archive "$commit" | tar -x -C "$temporary"
    mv "$temporary" "$snapshot"
fi
export HYDROSTATE_CODE_ROOT="$snapshot"
export HYDROSTATE_CODE_COMMIT="$commit"
exec /home/xiaoz/miniconda/envs/hydrostate/bin/python \
    "$snapshot/tools/submit_region.py" --root "$root" "$@"
