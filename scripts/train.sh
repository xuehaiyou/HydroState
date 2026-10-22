#!/bin/bash

set -euo pipefail

if [[ -n "${HYDROSTATE_CONDA_BASE:-}" ]]; then
    conda_base="$HYDROSTATE_CONDA_BASE"
elif [[ -f /home/xiaoz/miniconda/etc/profile.d/conda.sh ]]; then
    conda_base=/home/xiaoz/miniconda
else
    conda_base=/home/xiaoz/miniconda3
fi
source "$conda_base/etc/profile.d/conda.sh"
conda activate "${HYDROSTATE_CONDA_ENV:-hydrostate}"
cd "${HYDROSTATE_CODE_ROOT:-$(dirname "${BASH_SOURCE[0]}")/..}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

: "${HYDROSTATE_DATA_ROOT:?Set HYDROSTATE_DATA_ROOT to the uploaded snapshot directory}"
: "${HYDROSTATE_WORK_DIR:?Set HYDROSTATE_WORK_DIR to the experiment output directory}"
export HYDROSTATE_MODEL_PATH="${HYDROSTATE_MODEL_PATH:-/fossfs/xiaozhen/HydroState/pretrained/OlmoEarth-v1_2-Base}"
export HYDROSTATE_MANIFEST="${HYDROSTATE_MANIFEST:-$HYDROSTATE_DATA_ROOT/samples.parquet}"
test -s "$HYDROSTATE_MANIFEST"
test -s "$HYDROSTATE_MODEL_PATH/config.json"
test -s "$HYDROSTATE_MODEL_PATH/weights.pth"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1
precision="${HYDROSTATE_PRECISION:-bfloat16}"

# CUDA visibility reflects the GPUs granted to this job, not all GPUs on the node.
# Keep diagnostics on stderr so stdout contains only the detected count.
gpu_count="$(python - "${HYDROSTATE_GPUS_PER_NODE:-1}" "$precision" <<'PY'
import sys
import torch

count = torch.cuda.device_count()
if not torch.cuda.is_available() or count < 1:
    raise RuntimeError("No CUDA GPU is visible in this job; request a GPU with --gres")
if sys.argv[1] != "auto":
    try:
        requested = int(sys.argv[1])
    except ValueError as error:
        raise ValueError("HYDROSTATE_GPUS_PER_NODE must be 'auto' or a positive integer") from error
    if requested < 1:
        raise ValueError("HYDROSTATE_GPUS_PER_NODE must be 'auto' or a positive integer")
    if count != requested:
        raise RuntimeError(
            f"Visible {count} GPUs, explicitly requested {requested}; "
            "unset HYDROSTATE_GPUS_PER_NODE to detect automatically, or match --gres"
        )
for index in range(count):
    with torch.cuda.device(index):
        print(index, torch.cuda.get_device_name(index),
              torch.cuda.get_device_properties(index).total_memory, file=sys.stderr)
        if sys.argv[2] == "bfloat16":
            if not torch.cuda.is_bf16_supported(including_emulation=False):
                raise RuntimeError(
                    "GPU lacks native BF16; set HYDROSTATE_PRECISION=float16 and rerun the smoke test"
                )
print("PyTorch:", torch.__version__, "CUDA runtime:", torch.version.cuda, file=sys.stderr)
print(count)
PY
)"

config="${HYDROSTATE_CONFIG:-configs/hydrostate_mtlora.py}"
if [[ "$gpu_count" != 1 ]]; then
    echo 'Current retrieval training requires exactly one visible GPU.' >&2
    exit 2
fi
printf 'Detected 1 visible GPU; launching single-GPU training.\n'
exec python tools/train.py "$config" \
    --launcher none --cfg-options "optim_wrapper.dtype=$precision" "$@"
