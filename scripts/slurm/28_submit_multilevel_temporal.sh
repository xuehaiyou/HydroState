#!/bin/bash
# Reuse the verified ecosystem statistics; leave existing training jobs running.
set -euo pipefail
export HYDROSTATE_CODE_ROOT="${HYDROSTATE_CODE_ROOT:-/home/xiaoz/HydroState}"
cd "$HYDROSTATE_CODE_ROOT"
git diff --quiet HEAD -- || { echo "Commit source changes before submitting." >&2; exit 1; }
export HYDROSTATE_CODE_COMMIT="$(git rev-parse HEAD)"
export HYDROSTATE_CONFIG=configs/hydrostate_ecosystem_multilevel_temporal_hpc.py
export HYDROSTATE_RUN_PREFIX=ecosystem_multilevel_temporal
mkdir -p /scr/u/xiaoz/HydroState/logs /scr/u/xiaoz/HydroState/submissions
record="${HYDROSTATE_SUBMISSION_RECORD:-/scr/u/xiaoz/HydroState/submissions/ecosystem_multilevel_temporal_${HYDROSTATE_CODE_COMMIT}.txt}"
(set -o noclobber; printf 'CODE_COMMIT=%s\nCONFIG=%s\n' \
    "$HYDROSTATE_CODE_COMMIT" "$HYDROSTATE_CONFIG" > "$record")
export HYDROSTATE_SOURCE_HASHES="$record.source.sha256"
git ls-files -z | xargs -0 sha256sum > "$HYDROSTATE_SOURCE_HASHES"
source scripts/slurm/ecosystem_env.sh
test -s "$HYDROSTATE_TARGET_STATS"
python - <<'PY'
import os
from mmengine.config import Config
from hydrostate.cli import configure_target_statistics

cfg = Config.fromfile(os.environ["HYDROSTATE_CONFIG"])
configure_target_statistics(cfg, training=True)
assert cfg.model.type == "MultiLevelTemporalHydroStateModel"
assert cfg.model.encoder.out_indices == [3, 7, 11]
assert cfg.model.adapter.enabled and len(cfg.model.tasks) == 16
assert cfg.target_time_queries and cfg.label_packs[1].max_uncertainty.lst == 2.0
print("Multilevel temporal model: all 16 tasks and frozen training statistics validated")
PY
train=$(sbatch --parsable --job-name=hs_multilevel_temporal \
    scripts/slurm/26_train_ecosystem_temporal.sh)
train=${train%%;*}
printf 'TRAIN_JOB=%s\n' "$train" | tee -a "$record"
squeue -j "$train" -o '%.18i %.30j %.10T %.30R'
