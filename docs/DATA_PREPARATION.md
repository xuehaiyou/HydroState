# Global GEE sampled-data workflow

The active pipeline is `hydrostate/data/gee_sampling.py`, launched through
`scripts/run_gee_sampling.sh`. It reads all satellite imagery and product labels
from GEE. The former local coverage and station tools remain optional legacy tools.

## Environment and launch

Use the project's Python 3.12 environment:

```bash
pip install -e ".[geo,gee,dev]"
earthengine authenticate
export GEE_PROJECT=your-registered-project
export OUTPUT_DIR=/mnt/d/Hydrostate/samples/batch_seed43_n10000
export SEED=43
export CANDIDATE_DIR=/mnt/d/Hydrostate/candidates
# Authorized-user OAuth JSON with Drive scope, not the Earth Engine token file.
export DRIVE_TOKEN=/path/to/drive-oauth.json
bash scripts/run_gee_sampling.sh --plan-only
bash scripts/run_gee_sampling.sh
```

The Earth Engine account needs access to the private climate asset and a registered
project. Drive requires separately provisioned authorized-user OAuth credentials
with permission to list, download and delete this account's GEE exports. Tokens
are never written into the run configuration. Supply START_YEAR, END_YEAR, SAMPLES,
SEED, BATCH_SIZE, CONCURRENCY and PYTHON as environment variables if needed.

Default period: 2019–2024. Each input is 120×120 at 10 m. Four 16-day slots cover
[t−63,t] inclusive; each sensor contributes at most one scene per slot. Selection
uses window-valid fraction (default 0.5), proximity to the slot's latest date, and
consistent S1 relative orbit where available. At least two slots must be present;
the latest input must be within 3 days of the target DW observation.

## Reuse a shared candidate pool

`scripts/run_gee_sampling.sh` defaults to `/mnt/d/Hydrostate/candidates` for the
shared pool and `/mnt/d/Hydrostate/samples/batch_seed${SEED}_n${SAMPLES}` for each sample run.
The defaults are 10,000 samples and selection seed 43, creating a separate dataset
from the original seed-42 run in `/mnt/d/Hydrostate/samples`. Explicitly set
`OUTPUT_DIR=/mnt/d/Hydrostate/samples SEED=42` to resume that original run.
`CANDIDATE_SEED` (default 42) controls pool generation; `SEED` controls candidate
ranking and target dates. `SPLIT_SEED` (shell default 42) independently controls
geographic split assignment. The Python CLI defaults its split seed to `--seed`
when `--split-seed` is omitted, preserving legacy behavior. Changing
sample count, year range or `SEED` does not regenerate the shared spatial blocks.
Changing the climate source, candidate seed, strategy or candidates per class
requires a different pool directory. `pool_config.json` enforces this contract.
Each run keeps its own selections, sample plan, export state, downloads and Zarr.

```bash
# After the current global candidate generation finishes:
bash scripts/run_gee_sampling.sh  # New independent 10,000-sample seed-43 run
SAMPLES=30000 SEED=42 bash scripts/run_gee_sampling.sh
SAMPLES=50000 SEED=123 bash scripts/run_gee_sampling.sh
```

The current producer continues at `/mnt/d/Hydrostate/samples` unchanged.
`/mnt/d/Hydrostate/candidates` links to its `samples/candidates`, and
`/mnt/d/Hydrostate/samples_10000` links to `samples`. Neither link copies or moves
files. Do not delete the original `samples` directory: these links use its data.
New code cannot replace a running Python process's loaded functions. New runs
refuse to supplement an incomplete pool while the original producer owns it;
after all 12,960 block files exist, they can reuse it while the original run
continues selecting/exporting its samples. New builders also use a shared lock.

Different runs may overlap unless prior manifests are explicitly excluded.
Keep `SPLIT_SEED=42` when combining with the original run; changing it changes
geographic assignments and the merger rejects inconsistent splits. Requested
counts still depend on candidate coverage, quality and per-location date limits.

## Collect an additional batch and publish a combined Zarr dataset

```bash
bash scripts/run_gee_expansion.sh
```

This workflow collects 10,000 **additional** samples with `SEED=43` and
`SPLIT_SEED=42`. It reads the completed original manifest at
`/mnt/d/Hydrostate/samples/samples.parquet` and excludes existing window/month
neighborhoods while respecting the combined maximum of four dates per location.
It uses the shared candidate pool and writes the new batch to
`/mnt/d/Hydrostate/samples/batch_seed43_n10000`. Zarr schema v3 is unchanged.
The old data stay at their original location; later batches use their own
`batch_seed<seed>_n<count>` subdirectories under `samples`.

After sampling finishes, the workflow validates both manifests, geographic splits,
window/date duplicates, spacing, array shapes/dtypes, completion markers and
preprocessing metadata. It copies retained Zarr stores, without re-encoding,
into `/mnt/d/Hydrostate/samples/merged_v1/zarr` and publishes relative paths in
`/mnt/d/Hydrostate/samples/merged_v1/samples.parquet`. `merge_report.json` records
source hashes, duplicates removed and split/climate/frequency/month counts.
This is a self-contained directory for training or HPC transfer. Copies require
additional disk space. The directory is published only after validation and copy
complete; existing output versions are never overwritten.

Repeat the same expansion command to resume an interrupted sampling run.
Exclusion manifests are pinned by content hash; do not change them while a batch
is running. If sampling has completed but merge failed, run only the merger:

```bash
/home/xiaoz/miniconda3/envs/hydrostate/bin/python -m hydrostate.data.merge_samples \
  --source /mnt/d/Hydrostate/samples/samples.parquet \
  --source /mnt/d/Hydrostate/samples/batch_seed43_n10000/samples.parquet \
  --split-seed 42 --copy-zarr --output /mnt/d/Hydrostate/samples/merged_v1
```

To collect a subsequent batch after `merged_v1` is published, exclude that entire
combined manifest and publish a new version:

```bash
SEED=44 OLD_MANIFEST=/mnt/d/Hydrostate/samples/merged_v1/samples.parquet \
MERGED_DIR=/mnt/d/Hydrostate/samples/merged_v2 bash scripts/run_gee_expansion.sh
```

For training, set the manifest to `merged_v1/samples.parquet` and data root to
`merged_v1`; train/validation/test loaders select their respective `split` rows.

## Sampling

Climate classes come from
`users/xiaozhen6666666/cloud/Beck_KG_V1_present_0p0083`; use `--climate-band` if
needed. The default is its first band, checked at startup. Positive integral
classes are eligible. JRC GSW1.4 occurrence has seven bins:
0, (0,20], (20,40], (40,60], (60,80], (80,100), and 100, encoded as 0–6.
The zero bin includes observed land without recorded water; the 100 bin captures
historically permanent water. At least one non-no-data classification in
JRC/GSW1_4/YearlyHistory is required, so unobserved pixels are not labeled dry.
The YearlyHistory `waterClass` meanings are documented in the
[GEE catalog](https://developers.google.com/earth-engine/datasets/catalog/JRC_GSW1_4_YearlyHistory).
Climate coverage remains required for every bin, including permanent water.
This is a balanced training sample, not an area-representative global census.

Seven frequency bins receive equal quotas (1428–1429 each for 10,000 samples),
then year-month quotas are balanced.
Within each month/bin, half of climate allocation is equal and half is proportional
to eligible area. Candidate pools use 30 m EPSG:6933 sampling (explicit WKT1 projection for GEE) in cached 2-degree
blocks, within -60 to 84 degrees latitude (optical-product coverage). Per-block
class counts weight the candidate ordering, avoiding equal weighting of sparse
and dense blocks. Before 30 m sampling/statistics, each block is screened for
positive integral climate classes at the climate raster's native resolution.
A 2 km boundary buffer conservatively retains coastal and boundary pixels.
Blocks with no climate coverage are cached as skipped, and never request the
expensive 30 m JRC sample/histogram operations. Logs distinguish
`skipped=no climate coverage` from processed blocks with zero candidates.
This two-stage finite candidate pool is an approximation to
full-population sampling; do not treat its acceptance distribution as an exact
inclusion probability. Candidate generation is a substantial one-time global job.

5-degree geographic blocks determine 70/15/15 train/validation/test membership;
12-km boundary buffers reduce shared coarse-label leakage. Same-month overlapping
windows are rejected and each location is used at most four times. Spatial sample
selection and quality checks are cached. Deficits are logged and planning fails
unless `--allow-shortfall` explicitly accepts the partial plan. Increase
`--candidates-per-class` in a new output directory to enlarge the pool.

## Labels and grid

The actual target t is a valid DW observation within the assigned month.
DW label==0 is saved as binary water, with its mask; flooded vegetation is not
included. SMAP AM uses nearest valid observation within ±3 days and records its
signed date offset. SMAP 005 is used before 2023-12-04, 006 thereafter.
PML V2.2a ET is multiplied by 0.01 (mm/day), with the containing 8-day interval
recorded (year-end intervals are truncated). It is not an instantaneous daily ET.
The PML collection has no catalogued QC band; its mask is retained.

All arrays are nearest-neighbor sampled onto a fixed 10 m grid, with an origin
aligned to 30 m. Optical values retain S2 harmonized integers/Landsat L1 DN until
local official OlmoEarth normalization. S1 is GEE IW GRD dB, not RTC.

At training time the decoder/head outputs 40×40 at 30 m:

- Water: average binary DW in 3×3 blocks; all nine labels must be valid.
- ET: aggregate predictions with true polygon-intersection area weights to each
  complete native PML cell inside the window. Read the actual ET band projection
  from GEE, rather than assuming a 500 m grid. Exclude partial cells and cells with
  unobserved contributing predictions. The target is the containing 8-day mean
  daily ET, not instantaneous ET at t.
- SMAP: compare the mean of observed predictions with the nearest-neighbor label
  at the original window center. This is low-weight window-scale weak supervision,
  not full native SMAP support. No windows are linked.

An output pixel is observed when all its nine 10 m subpixels have at least one
valid sensor observation across the four slots. Water also requires this mask.
Losses use Huber on scaled values: reference scales water=1, SM=0.1 m³/m³,
ET=3 mm/day; task weights are 1, 0.25, 1. These are configurable initial
hyperparameters, not dataset-estimated standard deviations or tuned optima.
Evaluation uses the same spatial supports, in original physical units; its combined
score uses the same reference scales. Keep model/evaluator scales consistent if changed.
Station losses use the corresponding variable scale as well.

To reuse existing verified downloads without exporting imagery again:

```bash
GEE_PROJECT=amplifier3 OUTPUT_DIR=outputs/gee_multiscale_pilot \
  bash scripts/run_gee_sampling.sh \
  --reingest-from outputs/gee_pilot --reingest-from outputs/gee_pilot_sm
```

This reads only missing PML grid metadata from GEE, validates existing TIFF MD5/size,
and writes new v3 shards and `samples.parquet`. Drive OAuth is not required for this
conversion. Point the training/validation dataset at the new manifest and use
`hydro_collate` (already configured). Original downloads and shards are retained.
Use a new output directory for new production exports too; old run state is not
silently upgraded.

## Transfers and restart

One Float32 multiband GeoTIFF per sample; timestamps are stored separately in
metadata. Defaults: 32 samples per batch, at most two pending exports. A unique
Drive folder is derived from the output directory. Completed files are downloaded
with size/MD5 validation, checked for band order and georeferencing, normalized,
and written as immutable one-row input/label Zarr pairs. The local training
manifest is published before permanent deletion of the batch's registered Drive
file IDs. Other Drive files and local GeoTIFFs are retained.

SQLite stores task IDs, file IDs and commit/cleanup state. A directory lock blocks
concurrent writers. Network downloads retry individual requests; failed exports
retry up to three times, then stop with their error. Sampling configuration changes require
a new output directory. Export task submission persists its request ID first, then replaces it with the
returned operation/task ID. Recovery reconciles named tasks before resubmission.
GEE can export a one-pixel outer border due to geometry rounding; ingestion reads
the exact manifest-aligned 120×120 window and rejects any misaligned raster.

Outputs: `sample_index.parquet`, `samples.parquet`, `sampling_report.json`,
`state.sqlite`, cached candidates/selections, downloads and `zarr/`.
Point training's manifest at OUTPUT_DIR/samples.parquet and data_root at OUTPUT_DIR.
