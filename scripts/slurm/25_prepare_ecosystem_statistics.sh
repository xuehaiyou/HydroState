#!/bin/bash
#SBATCH --job-name=hs_ecosystem_stats
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=c_foss_amd
#SBATCH --account=geog_geors
#SBATCH --qos=normal
#SBATCH --time=24:00:00
#SBATCH --mail-type=NONE
#SBATCH --output=/scr/u/xiaoz/HydroState/logs/%x-%j.out
#SBATCH --error=/scr/u/xiaoz/HydroState/logs/%x-%j.err
set -euo pipefail
source "${HYDROSTATE_CODE_ROOT:-/home/xiaoz/HydroState}/scripts/slurm/ecosystem_env.sh"
if [[ ! -f "$HYDROSTATE_TARGET_STATS" ]]; then
    python tools/data/compute_target_statistics.py --config "$HYDROSTATE_CONFIG" \
        --output "$HYDROSTATE_TARGET_STATS" --num-workers "${SLURM_CPUS_PER_TASK:-8}"
fi
# Existing statistics must match these exact tasks, packs and training identities.
python - <<'PY'
import os
from mmengine.config import Config
from hydrostate.cli import configure_target_statistics

cfg = Config.fromfile(os.environ["HYDROSTATE_CONFIG"])
configure_target_statistics(cfg, training=True)
assert len(cfg.model.tasks) == 16 and cfg.target_time_queries
assert cfg.label_packs[1].max_uncertainty.lst == 2.0
print("All 16 tasks: frozen training statistics validated", flush=True)
PY
