#!/bin/bash
#SBATCH --job-name=hs_shared_private_compose
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --partition=intel
#SBATCH --account=geog_geors
#SBATCH --qos=normal
#SBATCH --time=02:00:00
#SBATCH --mail-type=NONE
set -euo pipefail
: "${HYDROSTATE_RUN_ROOT:?Use tools/submit_shared_private.py}"
: "${HYDROSTATE_TARGET_STATS:?Frozen L2 statistics are required}"
source "${HYDROSTATE_CODE_ROOT:?}/scripts/slurm/ecosystem_env.sh"
checkpoint=$(python -c \
  'import os; from tools.submit_shared_private import best_checkpoint; print(best_checkpoint(os.environ["HYDROSTATE_RUN_ROOT"]+"/shared30", "hydro_score"))')
config="$HYDROSTATE_RUN_ROOT/shared30/hydrostate_shared_private.py"
python tools/compose_shared_private.py "$config" "$checkpoint" \
  "$HYDROSTATE_RUN_ROOT/staged_composed.pth" --task-bundles \
  "$HYDROSTATE_RUN_ROOT/bundles/water.pth" \
  "$HYDROSTATE_RUN_ROOT/bundles/soil_moisture.pth" \
  "$HYDROSTATE_RUN_ROOT/bundles/et.pth"
python tools/export_shared_private.py "$config" "$HYDROSTATE_RUN_ROOT/staged_composed.pth" \
  "$HYDROSTATE_RUN_ROOT/knowledge_v1.pth"
