#!/bin/bash
#SBATCH --job-name=hs_mtlora_prepare
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --partition=intel
#SBATCH --account=geog_geors
#SBATCH --qos=normal
#SBATCH --time=24:00:00
#SBATCH --mail-type=NONE
set -euo pipefail
cd "${HYDROSTATE_CODE_ROOT:?}"
sha256sum --check --status "${HYDROSTATE_SOURCE_HASHES:?}"
source "${HYDROSTATE_CONDA_BASE:-/home/xiaoz/miniconda}/etc/profile.d/conda.sh"
conda activate hydrostate
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 BLOSC_NTHREADS=1
python - <<'PY'
import json
import os
from pathlib import Path
from tools.data.collect_gee_dataset import verify_samples, verify_pack
from hydrostate.storage.packs import file_hash
from hydrostate.data.temporal_sampling import CENTERED

root = Path(os.environ['HYDROSTATE_DATA_ROOT'])
cfg = json.loads((root / 'collection.json').read_text())
state = json.loads((root / 'collection_state.json').read_text())
marker = json.loads((root / '_COLLECTION_SUCCESS.json').read_text())
assert state['complete'] and marker['completed_at']
assert cfg['output_dir'] == str(root) and cfg['samples'] == 20000
assert cfg['temporal']['temporal_sampling'] == CENTERED
assert cfg['input_profile'] == 'olmoearth_l1_v1'
table = verify_samples(cfg)
assert set(table.split) == {'train', 'validation', 'test'}
assert marker['manifest_sha256'] == state['manifest_sha256'] == file_hash(root / 'samples.parquet')
for name in ('ecosystem_v1', 'lst_observation_v1'):
    verify_pack(root, name, table)
print('Verified centered 4 x 8 days, 20000 samples and complete label packs', flush=True)
PY
if [[ ! -f "${HYDROSTATE_TARGET_STATS:?}" ]]; then
    python tools/data/compute_target_statistics.py --config configs/hydrostate_mtlora.py \
        --output "$HYDROSTATE_TARGET_STATS" --num-workers 4
fi
python - <<'PY'
import json
import os
from pathlib import Path
from mmengine.config import Config
from hydrostate.cli import configure_target_statistics
from hydrostate.datasets.zarr_dataset import HydroStateZarrDataset
from hydrostate.storage.packs import file_hash

root = Path(os.environ['HYDROSTATE_RUN_ROOT'])
base = None
for model, rank in [('mtlora', 4), ('mtlora_shared', 0)]:
    cfg = Config.fromfile(f'configs/hydrostate_{model}.py')
    configure_target_statistics(cfg, training=True)
    assert cfg.model.encoder.private_rank == rank and cfg.model.encoder.shared_rank == 8
    assert cfg.target_time_queries and cfg.input_profile == 'olmoearth_l1_v1'
    assert cfg.train_dataloader.batch_size == 2 and cfg.optim_wrapper.accumulative_counts == 8
    if base is not None:
        assert cfg.train_dataloader.dataset == base.train_dataloader.dataset
    else:
        for key in ('train_dataloader', 'val_dataloader', 'test_dataloader'):
            ds = HydroStateZarrDataset(**{k: v for k, v in cfg[key].dataset.items() if k != 'type'})
            for index in range(min(3, len(ds))):
                sample = ds[index]
                assert sample['inputs']['sensor_valid'].shape == (4, 3)
                assert set(sample['inputs']['target_time']) == {'water', 'soil_moisture', 'et'}
    cfg.work_dir = str(root / ('lora' if rank == 0 else 'mtlora'))
    cfg.dump(str(root / 'provenance' / f'{model}.py'))
    base = cfg
report = dict(manifest_sha256=file_hash(os.environ['HYDROSTATE_MANIFEST']),
              statistics_sha256=file_hash(os.environ['HYDROSTATE_TARGET_STATS']),
              temporal_sampling='centered_8d_v1', samples=20000)
(root / 'provenance' / 'ready.json').write_text(json.dumps(report, indent=2) + '\n')
print('Both model data contracts and representative 4-slot reads passed', flush=True)
PY
