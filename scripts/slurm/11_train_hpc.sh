#!/bin/bash
#SBATCH --job-name=hs_hpc
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --account=geog_geors
#SBATCH --partition=c_foss_gpu
#SBATCH --qos=gpu
#SBATCH --mem-per-cpu=5G
#SBATCH --gres=gpu:4
#SBATCH --time=96:00:00
#SBATCH --output=/fossfs/xiaozhen/HydroState/logs/%x-%j.out

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
cd "${HYDROSTATE_CODE_ROOT:-/home/xiaoz/HydroState}"

: "${HYDROSTATE_DATA_ROOT:?Set HYDROSTATE_DATA_ROOT to the uploaded snapshot directory}"
: "${HYDROSTATE_WORK_DIR:?Set HYDROSTATE_WORK_DIR to the experiment output directory}"
export HYDROSTATE_MODEL_PATH="${HYDROSTATE_MODEL_PATH:-/fossfs/xiaozhen/HydroState/pretrained/OlmoEarth-v1_2-Base}"
export HYDROSTATE_MANIFEST="${HYDROSTATE_MANIFEST:-$HYDROSTATE_DATA_ROOT/samples.parquet}"
test -s "$HYDROSTATE_MANIFEST"
test -s "$HYDROSTATE_MODEL_PATH/config.json"
test -s "$HYDROSTATE_MODEL_PATH/weights.pth"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1
gpu_count="${HYDROSTATE_GPUS_PER_NODE:-4}"
precision="${HYDROSTATE_PRECISION:-bfloat16}"

python - "$gpu_count" "$precision" <<'PY'
import sys
import torch

requested = int(sys.argv[1])
assert requested > 0
assert torch.cuda.is_available(), "CUDA is unavailable in this GPU job"
assert torch.cuda.device_count() == requested, (
    f"Allocated {torch.cuda.device_count()} GPUs, requested {requested}; "
    "match --gres and HYDROSTATE_GPUS_PER_NODE"
)
for index in range(requested):
    with torch.cuda.device(index):
        print(index, torch.cuda.get_device_name(index), torch.cuda.get_device_properties(index).total_memory)
        if sys.argv[2] == "bfloat16":
            assert torch.cuda.is_bf16_supported(including_emulation=False), (
                "GPU lacks native BF16; set HYDROSTATE_PRECISION=float16 and rerun the smoke test"
            )
print("PyTorch:", torch.__version__, "CUDA runtime:", torch.version.cuda)
PY

exec torchrun --standalone --nnodes=1 --nproc_per_node="$gpu_count" \
    tools/train.py "${HYDROSTATE_CONFIG:-configs/hydrostate_hpc.py}" \
    --launcher pytorch --cfg-options "optim_wrapper.dtype=$precision" "$@"
