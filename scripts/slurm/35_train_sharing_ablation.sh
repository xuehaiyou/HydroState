#!/bin/bash
#SBATCH --job-name=hs_sharing_ablation
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=l40s
#SBATCH --account=geog_geors
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=GPU_MEM:48GB
#SBATCH --time=96:00:00
#SBATCH --mail-type=NONE
set -euo pipefail
stage="${1:?Use tools/submit_sharing_ablation.py}"
[[ "$stage" =~ ^(shared|shared_private|independent_(water|soil_moisture|et))_s[0-9]+$ ]]
: "${HYDROSTATE_RUN_ROOT:?Missing run root}"
source "${HYDROSTATE_CODE_ROOT:?}/scripts/slurm/ecosystem_env.sh"
sha256sum --check --status "${HYDROSTATE_ASSET_HASHES:?}"
export HYDROSTATE_CONFIG="$HYDROSTATE_RUN_ROOT/configs/$stage.py"
export HYDROSTATE_WORK_DIR="$HYDROSTATE_RUN_ROOT/$stage"
export HYDROSTATE_PRECISION=bfloat16 HYDROSTATE_GPUS_PER_NODE=1
mkdir -p "$HYDROSTATE_WORK_DIR/provenance"
printf '%s\n' "$HYDROSTATE_CODE_COMMIT" > "$HYDROSTATE_WORK_DIR/provenance/code_commit.txt"
date -u +%FT%TZ > "$HYDROSTATE_WORK_DIR/provenance/train_started_utc.txt"
started=$(date +%s)
bash scripts/slurm/11_train_hpc.sh
printf '%s\n' "$(( $(date +%s) - started ))" \
  > "$HYDROSTATE_WORK_DIR/provenance/training_wall_seconds.txt"
python tools/evaluate_sharing_ablation.py "$HYDROSTATE_CONFIG" "$HYDROSTATE_WORK_DIR"
