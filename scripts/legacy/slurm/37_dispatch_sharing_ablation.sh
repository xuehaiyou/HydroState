#!/bin/bash
#SBATCH --job-name=hs_sharing_dispatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --partition=intel
#SBATCH --account=geog_geors
#SBATCH --qos=normal
#SBATCH --time=00:10:00
#SBATCH --mail-type=NONE
set -euo pipefail
source "${HYDROSTATE_CODE_ROOT:?}/scripts/legacy/slurm/ecosystem_env.sh"
python tools/legacy/dispatch_sharing_ablation.py "$HYDROSTATE_RUN_ROOT/submission.json"
