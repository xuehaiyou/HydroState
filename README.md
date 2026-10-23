# HydroState

Joint retrieval of water, snow/ice, land fractions, land soil moisture, and
water/ice/land ET with OlmoEarth v1.2 Base. Current inputs are **five 8-day mean
composites** of Sentinel-1, Sentinel-2 and Landsat, collected from GEE.
Outputs are 30 m; product losses respect each label's observation support.

## Start here

- [Scientific and sampling specification](docs/RETRIEVAL_PIPELINE.md)
- [Collect, prepare, train and validate](docs/DATA_PREPARATION.md)
- [Input specifications and independent label packs](docs/EXTENDING_VARIABLES.md)
- [MTLoRA and shared-LoRA architectures](docs/MTLORA.md)
- [Repository responsibilities](docs/REPOSITORY_STRUCTURE.md)
- [Implementation and validation status](docs/IMPLEMENTATION_STATUS.md)
- [All documentation](docs/README.md)

## Environment

Use Python 3.12 and an activated `hydrostate` environment:

```bash
conda create -n hydrostate python=3.12 -y
conda activate hydrostate
# Use the PyTorch wheel appropriate to the host CUDA driver.
pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.9.1 torchvision==0.24.1
pip install -e '.[geo,gee,dev]'
```

Set `HYDROSTATE_MODEL_PATH` to an existing OlmoEarth-v1_2-Base bundle containing
`config.json` and `weights.pth`. The collection process needs authorized Earth
Engine access. Raw local SAFE/Landsat directories are not part of the current
collection workflow.

## Current commands

Run from the repository root. Each stage is explicit; collection does not launch training.

```bash
export HYDROSTATE_DATA_ROOT=/path/to/samples/retrieval_5x8_native_v2
export HYDROSTATE_CANDIDATE_ROOT=/path/to/candidates/retrieval_5x8_v1
export HYDROSTATE_MODEL_PATH=/path/to/OlmoEarth-v1_2-Base

bash scripts/run_gee_sampling.sh --dry-run
bash scripts/run_gee_sampling.sh
python tools/validate_zarr.py "$HYDROSTATE_DATA_ROOT"
python tools/data/compute_target_statistics.py --config configs/hydrostate_mtlora.py   --output "$HYDROSTATE_DATA_ROOT/target_stats.json" --num-workers 4
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/hydrostate_mtlora.py
CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/hydrostate_mtlora.py CHECKPOINT.pth
```

Use `configs/hydrostate_mtlora_shared.py` for the shared-LoRA control on identical
data. [Current Slurm scripts](scripts/slurm) request one GPU and at most six CPUs;
see the preparation guide for environment variables and release checks.

## One current contract

[`hydrostate/data/retrieval/contract.py`](hydrostate/data/retrieval/contract.py)
defines the version, tasks, input profile and array shapes. Collection, readers,
training configs and release checks use this definition. Current entry points
reject old sample contracts; they do not convert four-slot or variable-pack data.

Historical model configurations and commands are kept under
[`configs/experiments/legacy`](configs/experiments/legacy),
[`tools/legacy`](tools/legacy), [`scripts/legacy`](scripts/legacy), and
[`docs/history`](docs/history). They preserve experiment context and are not
current data preparation instructions. Model implementations remain available
for controlled architecture comparisons.
