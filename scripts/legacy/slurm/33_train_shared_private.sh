#!/bin/bash
#SBATCH --job-name=hs_shared_private
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
stage="${1:?Select joint50, shared30, water, soil_moisture or et}"
shift
: "${HYDROSTATE_RUN_ROOT:?Use tools/legacy/submit_shared_private.py}"
: "${HYDROSTATE_TARGET_STATS:?Frozen L2 statistics are required}"
source "${HYDROSTATE_CODE_ROOT:?}/scripts/legacy/slurm/ecosystem_env.sh"
export HYDROSTATE_WORK_DIR="$HYDROSTATE_RUN_ROOT/$stage"
export HYDROSTATE_PRECISION=bfloat16 HYDROSTATE_GPUS_PER_NODE=1
case "$stage" in
  joint50) export HYDROSTATE_CONFIG=configs/experiments/legacy/hydrostate_shared_private_joint50.py ;;
  shared30) export HYDROSTATE_CONFIG=configs/experiments/legacy/hydrostate_shared_private.py ;;
  water|soil_moisture|et)
    export HYDROSTATE_CONFIG=configs/experiments/legacy/hydrostate_shared_private_specialize.py
    export HYDROSTATE_SPECIALIZE_TASK="$stage"
    HYDROSTATE_SP_CHECKPOINT=$(python -c \
      'import os; from tools.legacy.submit_shared_private import best_checkpoint; print(best_checkpoint(os.environ["HYDROSTATE_RUN_ROOT"]+"/shared30", "hydro_score"))')
    export HYDROSTATE_SP_CHECKPOINT
    ;;
  *) echo "Unsupported stage: $stage" >&2; exit 2 ;;
esac
mkdir -p "$HYDROSTATE_WORK_DIR/provenance"
cp "$HYDROSTATE_TARGET_STATS" "$HYDROSTATE_WORK_DIR/provenance/target_statistics.json"
printf '%s\n' "$HYDROSTATE_CODE_COMMIT" > "$HYDROSTATE_WORK_DIR/provenance/code_commit.txt"
printf '%s\n' "${HYDROSTATE_SP_CHECKPOINT:-foundation initialization}" \
  > "$HYDROSTATE_WORK_DIR/provenance/initialization.txt"
# Uses actual allocated CUDA visibility, BF16 checks and the existing MMEngine launcher.
bash scripts/legacy/slurm/11_train_hpc.sh "$@"
if [[ "$stage" != joint50 && "$stage" != shared30 ]]; then
  checkpoint=$(python -c \
    'import os; from tools.legacy.submit_shared_private import best_checkpoint; print(best_checkpoint(os.environ["HYDROSTATE_WORK_DIR"], os.environ["HYDROSTATE_SPECIALIZE_TASK"]+"_rmse"))')
  python tools/legacy/export_shared_private.py \
    "$HYDROSTATE_WORK_DIR/hydrostate_shared_private_specialize.py" \
    "$checkpoint" "$HYDROSTATE_RUN_ROOT/bundles/$stage.pth" --tasks "$stage"
fi
