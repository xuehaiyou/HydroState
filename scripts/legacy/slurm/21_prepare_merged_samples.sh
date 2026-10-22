#!/bin/bash
#SBATCH --job-name=hs_merged_prepare
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=c_foss_amd
#SBATCH --account=geog_geors
#SBATCH --qos=normal
#SBATCH --time=24:00:00
#SBATCH --output=/home/xiaoz/HydroState/logs/%x-%j.out
#SBATCH --error=/home/xiaoz/HydroState/logs/%x-%j.err
set -euo pipefail
cd /home/xiaoz/HydroState
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python=/home/xiaoz/miniconda/envs/hydrostate/bin/python
source_root=/fossfs/xiaozhen/HydroState/samples_v1
export HYDROSTATE_DATA_ROOT=/scr/u/xiaoz/HydroState/samples_merged_v1
export HYDROSTATE_MANIFEST="$HYDROSTATE_DATA_ROOT/manifests/merged_v1_n20000/samples.parquet"
export HYDROSTATE_TARGET_STATS="${HYDROSTATE_MANIFEST%/*}/target_stats.json"
source_manifest="$source_root/manifests/merged_v1_n20000/samples.parquet"
mkdir -p "${HYDROSTATE_MANIFEST%/*}"
exec 9>"$HYDROSTATE_DATA_ROOT/.prepare.lock"
flock -n 9
"$python" - "$source_manifest" "$HYDROSTATE_MANIFEST" <<'PY'
import hashlib, os, sys
from pathlib import Path, PurePosixPath
import pandas as pd
source, target = map(Path, sys.argv[1:])
content = source.read_bytes()
expected = '5e69b409de2fe90138acd714d915ef401364a57d0f4c6d9d7309f58c39f55650'
assert hashlib.sha256(content).hexdigest() == expected, 'Published manifest changed'
table = pd.read_parquet(source)
assert len(table) == 20000 and table.sample_id.is_unique
assert table.split.value_counts().to_dict() == {'train':14967,'test':2814,'validation':2219}
if target.exists():
    assert target.read_bytes() == content, 'Existing scratch manifest differs'
paths = sorted(set(table.input_shard) | set(table.label_shard))
for value in paths:
    path = PurePosixPath(value)
    assert not path.is_absolute() and '..' not in path.parts and path.parts[0] == 'zarr'
(target.parent / 'zarr-files.txt').write_text(''.join(p+'\n' for p in paths))
staged = target.with_suffix('.parquet.part')
staged.write_bytes(content)
os.replace(staged, target)
print('Pinned manifest:', len(table), table.split.value_counts().to_dict(), flush=True)
PY
rsync -rlt --partial --stats --info=progress2 \
    --files-from="${HYDROSTATE_MANIFEST%/*}/zarr-files.txt" \
    "$source_root/" "$HYDROSTATE_DATA_ROOT/"
# Compute fresh training-only moments; never reuse the 1,024-sample statistics.
if [[ ! -s "$HYDROSTATE_TARGET_STATS" ]]; then
    "$python" -u tools/legacy/data/compute_target_statistics.py \
        --manifest "$HYDROSTATE_MANIFEST" --data-root "$HYDROSTATE_DATA_ROOT" \
        --num-workers 8 --output "$HYDROSTATE_TARGET_STATS"
fi
"$python" - <<'PY'
import json, os
from pathlib import Path
from mmengine.config import Config
from hydrostate.cli import configure_target_statistics
cfg = Config.fromfile('configs/experiments/legacy/hydrostate_merged.py')
configure_target_statistics(cfg, training=True)
assert cfg.model.loss.type == 'UncertaintyWeightedNLL'
assert cfg.load_from is None and cfg.resume is False
stats = cfg.model.target_statistics
assert stats['training_samples'] == 14967
ready = Path(os.environ['HYDROSTATE_MANIFEST']).with_name('training_ready.json')
ready.write_text(json.dumps(dict(training_samples=14967,
    training_fingerprint=stats['training_fingerprint'],
    manifest_sha256=stats['manifest_sha256']), indent=2))
print('Scratch dataset and training-only statistics are ready', flush=True)
PY
