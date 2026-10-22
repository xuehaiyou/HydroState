#!/bin/bash
#SBATCH --job-name=hs_sharing_summary
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
source "${HYDROSTATE_CODE_ROOT:?}/scripts/legacy/slurm/ecosystem_env.sh"
sha256sum --check --status "${HYDROSTATE_ASSET_HASHES:?}"
python tools/legacy/summarize_sharing_ablation.py "$HYDROSTATE_RUN_ROOT/submission.json"
