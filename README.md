# HydroState

HydroState uses Sentinel-1, Sentinel-2, and Landsat time series with an
OlmoEarth v1.2 Base encoder to jointly retrieve water fraction, surface soil
moisture, and evapotranspiration. Training is driven by MMEngine; sampled and
aligned image windows are read from sharded Zarr stores.

## Paths

- Code: `/home/xiaoz/HydroState`
- Data, pretrained weights, and run outputs: `/fossfs/xiaozhen/HydroState`
- S1 source: `/fossfs/QianWang/Download/Sentinel1/tile`
- S2 source: `/fossfs/DATAPOOL/Sentinel2_L2`
- Landsat source: `/fossfs/DATAPOOL/Landsat_L2`

The raw sources are only read while creating sampled windows. Training reads
the Zarr stores and `samples.parquet` manifest.

## Environment

Use Python 3.12. The host Python 3.13 environment is intentionally not used.

```bash
conda create -n hydrostate python=3.12 -y
conda activate hydrostate

# Choose the PyTorch command that matches the cluster CUDA driver. For CUDA 12.8:
pip install --index-url https://download.pytorch.org/whl/cu128 \
  "torch>=2.7,<2.8" "torchvision>=0.22,<0.23"

cd /home/xiaoz/HydroState
pip install -e ".[geo,dev]"
```

The editable install above obtains the current official OlmoEarth and rslearn
repositories. To refresh them explicitly later, run:

```bash
pip install "git+https://github.com/allenai/olmoearth_pretrain.git"
pip install "git+https://github.com/allenai/rslearn.git"
```

Download and pin the v1.2 Base bundle:

```bash
hf download allenai/OlmoEarth-v1_2-Base \
  --local-dir /fossfs/xiaozhen/HydroState/pretrained/OlmoEarth-v1_2-Base
```

## Commands

```bash
python tools/validate_zarr.py configs/hydrostate_v1.py
python tools/train.py configs/hydrostate_v1.py
python tools/test.py configs/hydrostate_v1.py CHECKPOINT.pth
```

Distributed training:

```bash
torchrun --nproc_per_node=4 tools/train.py configs/hydrostate_v1.py --launcher pytorch
```

See `docs/DATA_SCHEMA.md` for the manifest and Zarr contract. Dataset creation
is deliberately separate from training so input shards remain immutable and a
new label family can be added without duplicating satellite imagery.

Local coverage indexing, station ingestion, and ready-to-submit SLURM jobs are
documented in `docs/DATA_PREPARATION.md`.

## Current implementation boundary

The MMEngine training path, sampled-Zarr contract, coverage indexing and station
ingestion are implemented. GEE sample-manifest generation and independent
source-to-Zarr producers are the remaining data-pipeline stages.
