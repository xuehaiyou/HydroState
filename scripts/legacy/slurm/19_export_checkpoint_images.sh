#!/bin/bash
#SBATCH --job-name=hs_export_images
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=l40s
#SBATCH --account=geog_geors
#SBATCH --qos=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --time=01:00:00
#SBATCH --mail-type=NONE
#SBATCH --output=/home/xiaoz/HydroState/logs/%x-%j.out
#SBATCH --error=/home/xiaoz/HydroState/logs/%x-%j.err
set -euo pipefail

usage() {
    cat <<'HELP'
Usage: sbatch scripts/legacy/slurm/19_export_checkpoint_images.sh RUN_DIR [NUM_SAMPLES]

Export WF/SM/ET comparisons from RUN_DIR/last_checkpoint using the saved config.
NUM_SAMPLES defaults to HYDROSTATE_VIS_NUM_SAMPLES, or 4.
RUN_DIR can also be supplied through HYDROSTATE_WORK_DIR.

Optional environment overrides:
  HYDROSTATE_EXPORT_CHECKPOINT  Checkpoint path (relative paths use RUN_DIR).
  HYDROSTATE_EXPORT_CONFIG      Saved config path (relative paths use RUN_DIR).
  HYDROSTATE_EXPORT_DIR         Output directory (relative paths use RUN_DIR).
  HYDROSTATE_VIS_ET_VMAX        ET color scale maximum in mm/day (default: 10).

Default images: RUN_DIR/checkpoint_visualizations/CHECKPOINT_NAME/epoch_000/*.png
The columns are 30 m prediction, supervision-scale prediction and ground truth.
HELP
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi
run_dir="${1:-${HYDROSTATE_WORK_DIR:-}}"
num_samples="${2:-${HYDROSTATE_VIS_NUM_SAMPLES:-4}}"
if [[ $# -gt 2 || -z "$run_dir" || ! "$num_samples" =~ ^[1-9][0-9]*$ ]]; then
    usage >&2
    printf 'Provide a training directory and a positive sample count.\n' >&2
    exit 2
fi
# Resolve the input directory before changing to the repository root.
run_dir="$(cd "$run_dir" && pwd -P)"
cd "${HYDROSTATE_CODE_ROOT:-/home/xiaoz/HydroState}"
source "${HYDROSTATE_CONDA_BASE:-/home/xiaoz/miniconda}/etc/profile.d/conda.sh"
conda activate "${HYDROSTATE_CONDA_ENV:-hydrostate}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
# These are trusted local MMEngine checkpoints, including training metadata.
# MMEngine 0.10 uses torch.load without an explicit weights_only argument.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

python - "$run_dir" "$num_samples" <<'PY'
import os
import sys
from pathlib import Path

from mmengine.config import Config
from mmengine.runner import Runner

from hydrostate.cli import configure_target_statistics
from hydrostate.engine import HydroVisualizationHook

run_dir = Path(sys.argv[1])
num_samples = int(sys.argv[2])


def resolve(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else run_dir / path


checkpoint_name = os.environ.get("HYDROSTATE_EXPORT_CHECKPOINT")
if not checkpoint_name:
    pointer = run_dir / "last_checkpoint"
    if not pointer.is_file():
        raise FileNotFoundError(f"No {pointer}; set HYDROSTATE_EXPORT_CHECKPOINT explicitly")
    checkpoint_name = pointer.read_text().strip()
if not checkpoint_name:
    raise ValueError("last_checkpoint is empty")
checkpoint = resolve(checkpoint_name)
if not checkpoint.is_file():
    raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")

config_name = os.environ.get("HYDROSTATE_EXPORT_CONFIG")
if config_name:
    config_path = resolve(config_name)
else:
    candidates = sorted(run_dir.glob("*.py"))
    if len(candidates) != 1:
        raise ValueError(
            f"Expected one saved .py config in {run_dir}, found {len(candidates)}; "
            "set HYDROSTATE_EXPORT_CONFIG to the training config"
        )
    config_path = candidates[0]

cfg = Config.fromfile(str(config_path))
configure_target_statistics(cfg, training=False)
if cfg.get("val_dataloader") is None or cfg.get("val_evaluator") is None:
    raise ValueError("The saved config must define validation data and an evaluator")

output = resolve(os.environ.get("HYDROSTATE_EXPORT_DIR") or
                 f"checkpoint_visualizations/{checkpoint.stem}")
cfg.work_dir = str(output)
cfg.load_from = str(checkpoint)
cfg.resume = False
cfg.launcher = "none"
for key in ("train_cfg", "train_dataloader", "optim_wrapper", "param_scheduler"):
    cfg[key] = None
cfg.default_hooks = dict(cfg.get("default_hooks") or {}, checkpoint=None)
cfg.log_processor = dict(cfg.get("log_processor") or {}, by_epoch=True)
hook = dict(
    type="HydroVisualizationHook",
    num_samples=num_samples,
    interval=1,
    out_dir=".",
    et_vmax=float(os.environ.get("HYDROSTATE_VIS_ET_VMAX", "10")),
)
HydroVisualizationHook(**{key: value for key, value in hook.items() if key != "type"})
cfg.custom_hooks = [hook]

print(f"Config: {config_path}")
print(f"Checkpoint: {checkpoint}")
print(f"Exporting up to {num_samples} fixed validation samples")
Runner.from_cfg(cfg).val()
print(f"Images: {output / 'epoch_000'}")
PY
