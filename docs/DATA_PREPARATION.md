# Global GEE sampled-data workflow

The active pipeline is `hydrostate/data/gee_sampling.py`, launched through
`scripts/run_gee_sampling.sh`. It reads all satellite imagery and product labels
from GEE. The former local coverage and station tools remain optional legacy tools.

## Add variables to existing samples

Use immutable input/label packs to extend an existing sample library. This does
not rerun sample selection or redownload the satellite inputs. A source-specific
producer first obtains arrays on the existing windows and observation times, from
GEE, local rasters, or another provider. The new importer consumes already aligned
arrays; it does not add automatic downloads for arbitrary GEE collection names.

For example, a terrain pack specification `terrain.json` is:

```json
{
  "kind": "input",
  "source": "Your elevation product, release and resampling recipe",
  "version": "1",
  "variables": {
    "elevation": {
      "shape": [1, 120, 120],
      "dtype": "float32",
      "units": "m",
      "temporal": "static",
      "grid": "input"
    }
  }
}
```

Prepare a Parquet records file with `sample_id` and `npz_path` columns. Each NPZ
contains `elevation` and `elevation_valid` arrays of the specified shape; paths
are relative to the records file. A sequence input also needs `<variable>_time`
as an integer UTC-epoch-seconds vector. IDs are copied from `samples.parquet`.

```bash
python tools/data/build_variable_pack.py \
  --manifest /path/to/dataset/samples.parquet \
  --records /path/to/aligned/terrain_records.parquet \
  --spec /path/to/terrain.json \
  --output /path/to/dataset/inputs/terrain_v1
```

To add temperature regression labels at the model's 30 m output grid, use
`kind="label"` and a variable specification such as
`"temperature": {"shape": [1,40,40], "units": "K", "grid": "output"}`.
The NPZ keys are `temperature` and `temperature_valid`. Spatial/time matching,
quality masks and unit conversion remain the producer's responsibility. Do not
turn coarse native products into fine-resolution truth by interpolation.

Producers can avoid intermediate NPZ files by calling the streaming writer directly:

```python
from hydrostate.storage import VariablePackWriter

with VariablePackWriter(
    "dataset/inputs/terrain_v1", "dataset/samples.parquet",
    kind="input", source="product and processing recipe", version="1",
    variables={"elevation": dict(shape=[1, 120, 120], units="m",
                                  temporal="static", grid="input")},
) as writer:
    # aligned_records yields existing sample IDs and arrays; missing pixels have false masks.
    for sample_id, elevation, valid in aligned_records:
        writer.write(sample_id, {"elevation": elevation, "elevation_valid": valid})
    writer.finalize()
```

`configs/hydrostate_extended.py` is a complete configuration example adding these
two packs. Set its dataset paths for your library. Shared top-level `tasks`,
`input_packs`, and `label_packs` are propagated by the training/statistics CLI to
the datasets, model and evaluators. Explicit `variables` selects which arrays to
read. Model heads must match the task names; `input_fusion.inputs` must match the
selected auxiliary inputs. For direct `Runner.from_cfg` use, call
`hydrostate.cli.configure_target_statistics` first, as the standard CLI does.

Compute a new frozen statistics file using the same configuration before training:

```bash
python tools/data/compute_target_statistics.py \
  --config configs/hydrostate_extended.py \
  --output /path/to/dataset/target_stats_extended_v1.json --num-workers 4

HYDROSTATE_TARGET_STATS=/path/to/dataset/target_stats_extended_v1.json \
  python tools/train.py configs/hydrostate_extended.py
```

Only training rows contribute statistics. A pack may omit samples; their new
labels are masked out while other tasks can still train. Adding/replacing a pack
requires a new statistics version. Keep published packs and statistics files
alongside the saved experiment configuration. See `DATA_SCHEMA.md` for supported
shapes, observation support, provenance checks and current adapter boundaries.

## Environment and launch

### New centered-eight-day library: satellites and all 16 tasks

Run the complete workflow from the repository with one command:

```bash
# Offline preview; no GEE calls or dataset directories are created.
bash scripts/run_gee_collection.sh --dry-run

# Collect and validate the complete new library.
bash scripts/run_gee_collection.sh
```

The MMEngine-style configuration is
`configs/data/collection/centered8d_16vars.py`. Defaults are 10,000 samples,
2019–2024, selection seed 43, and geographic split seed 42. It writes a new
subfolder `/mnt/d/Hydrostate/samples/centered8d_16vars_v1`. To change the count,
choose another output version at the same time:

```bash
bash scripts/run_gee_collection.sh \
  --samples 20000 \
  --output /mnt/d/Hydrostate/samples/centered8d_16vars_n20000_v1
```

For a background WSL run, keep the launcher log outside the new dataset folder:

```bash
mkdir -p outputs/collection
nohup bash scripts/run_gee_collection.sh \
  > outputs/collection/centered8d_16vars_v1.log 2>&1 < /dev/null &
```

The launcher uses `computePixels` for these small windows. It sequentially runs
satellite/base-label collection, ecosystem label collection, acquisition-time
LST extension, and two local validators. It never submits training jobs.
Landsat is **LC08/LC09 C2 T1 L1 DN**, S1 is GRD VV/VH sigma-zero dB, and S2 is
L2A SR Harmonized. These inputs match the current regional GEE L1 contract.
Downloading L2 surface temperature for the LST label does not change Landsat
input bands into L2 surface reflectance.

The 16 tasks are the nine DW cover fractions (water/WF, trees, grass, flooded
vegetation, crops, shrub and scrub, built, bare, snow and ice), SM, ET, LAI, GPP,
black-sky albedo, white-sky albedo, and LST. All are attempted for every sample;
missing observations keep false masks. The collector does **not** require all
three sensors or all 16 valid labels in each sample.

Inputs use the centered four 8-day slots, at least two distinct active slots,
50% scene/window validity, and no extra recency gate. Input acquisition times
and label instants/product intervals are preserved; coarse labels retain their
native spatial support. LST retains the existing <=2 K uncertainty threshold
and may use a paired input acquisition date.

```text
centered8d_16vars_v1/
  collection.json                frozen complete workflow configuration
  collection_sampling.json       frozen 4 x 8 day / >=2 slot selection policy
  config.json                    base sampler's frozen configuration
  sample_index.parquet            selected sample plan
  samples.parquet                 committed input/base-label manifest
  zarr/<sample_id>/inputs.zarr    S1, S2, Landsat, masks and real timestamps
  zarr/<sample_id>/labels.zarr    WF, SM and ET
  labels/ecosystem_v1/            nine cover classes + LAI/GPP/albedos/baseline LST
  labels/lst_observation_v1/      selected LST with its actual acquisition time
  labels/.*.work/                 resumable label download checkpoints
  logs/                          samples/ecosystem/lst/validation logs
  reports/                       local validation reports
  collection_state.json          active stage and pinned completed manifest
  _COLLECTION_SUCCESS.json       written after all stages and validators pass
```

Monitor the active stage in `collection_state.json` and read its matching log,
for example `tail -f .../logs/samples.log`. Re-run the identical command to
resume. The launcher verifies manifest identity and full pack sample-ID coverage,
skips completed downloads, and repeats the local validators. By default,
pixel-level audits inspect 64 samples; set `validation_limit = 0` in a separate
collection config for all-sample pixel audits. The LST training-support/backward
check remains limited to 64 samples. These are data checks, not model accuracy tests.
Changes to frozen configuration require a new folder. Existing datasets are
refused, and a lock prevents duplicate collection in the same output directory.
Only the spatial candidate pool is shared; old selected scenes and old labels
are not inherited.

After collection, `HYDROSTATE_DATA_ROOT` can point
`configs/hydrostate_ecosystem_temporal.py` at this library. Compute new
training-only target statistics before training; do not reuse merged_v1 statistics.

For direct HPC collection, use `configs/data/collection/centered8d_16vars_hpc.py`.
It collects **20,000 samples**, writes
`/scr/u/xiaoz/HydroState/samples/centered8d_16vars_n20000_v1`, and reads the
shared candidate pool at `/scr/u/xiaoz/HydroState/candidates`. Run
`scripts/run_gee_collection_io.sh` on `hpc2021-io1` or `hpc2021-io2`, with
`HYDROSTATE_CODE_ROOT`, `HYDROSTATE_SOURCE_HASHES`, and `HYDROSTATE_GEE_PYTHON`
set to the verified frozen release, its checksum list, and the existing GEE
interpreter. The wrapper restricts the process and its children to two CPUs,
nice 10 and one numerical-library thread. This network acquisition runs on an
IO node, outside Slurm; retain its hostname, PID, commit, output and log in the
submission record. Do not put it on offline compute nodes.

### Add ecosystem labels to the existing 20,000 merged_v1 samples

The producer uses `computePixels` directly and never changes base samples or
inputs. It reuses each sample's exact DW asset, PML GPP from the selected ET
asset when available (otherwise the containing interval), containing MCD15A3H
LAI, same-date MCD43A3 albedo, and same-UTC-date Landsat 8/9 C2 L2 surface
temperature. This baseline producer keeps the original same-date LST policy;
the separate observation-time extension below adds paired Landsat dates.

```bash
# Run tests and validate the 64-sample pilot; download it only if absent.
bash scripts/verify_gee_labels.sh

# Run all 20,000 samples, reusing completed pilot downloads.
WORKERS=8 bash scripts/run_gee_labels.sh
```

Defaults: `SAMPLE_ROOT=/mnt/d/Hydrostate/samples/merged_v1`, production
`OUTPUT_DIR=$SAMPLE_ROOT/labels/ecosystem_v1`, pilot
`PILOT_DIR=$SAMPLE_ROOT/labels/ecosystem_pilot_v2`, and shared
`WORK_DIR=$SAMPLE_ROOT/labels/.ecosystem_clear_v1.work`. These paths, `PYTHON`
and `GEE_PROJECT` are overridable environment variables. Re-run the same command
after interruption: a SQLite checkpoint preserves successful samples; failed
requests are retried and never turned into missing observations. A completed
pack is immutable; publishing to an existing directory is refused. The work
directory contains a reusable compressed cache, not training ZIP archives.

Published ordinary Zarr directories contain nine cover indicators, LAI, GPP,
black-sky and white-sky shortwave albedo, LST and LST uncertainty. Existing ET
and SM remain in the base library, making 16 configured prediction tasks.
`coverage.json` reports source coverage and separate 2/3/5 K sensitivity counts;
these comparison counts do not change the selected 2 K training cutoff.

- DW uses categorical `label`, not confidence probabilities, and retains a
  common mask. Each valid 30 m label requires nine valid 10 m subpixels.
- LAI uses `Lai * 0.1`, main algorithm with/without saturation, clear cloud state,
  and excludes extra-QC snow, high aerosol, cirrus, internal cloud and shadow.
- GPP uses `GPP * 0.01` gC/m²/day on its containing eight-day interval. It shares
  the PML model with ET; their relationship is not independent validation.
- Albedo uses both shortwave bands times 0.001 with full BRDF inversions. Its
  daily date represents a 16-day retrieval window, not an instantaneous value.
- LST uses `ST_B10 * 0.00341802 + 149` K, L2SP scenes, and excludes fill, cloud,
  dilated cloud, cirrus and shadow. Clear snow remains eligible. Store `ST_QA *
  0.01` K and exclude uncertainty **above 2 K** in training. The 30 m output
  value averages nearest-sampled 10 m contributors; all contributors must pass
  source masks, and their maximum uncertainty is retained conservatively.
  Landsat thermal information remains approximately 100 m, despite the 30 m grid.
- LAI/GPP/albedo use complete native footprints with equal-area polygon weights.
  Missing/partial cells cannot supervise a cropped or upsampled pseudo-observation.

After the full pack is published, validate it and create separate training-only
statistics. The configuration defaults to the 2 K cutoff and this new statistics
file; it does not reuse the original three-task statistics:

```bash
python tools/data/validate_gee_labels.py \
  --manifest /mnt/d/Hydrostate/samples/merged_v1/samples.parquet \
  --pack /mnt/d/Hydrostate/samples/merged_v1/labels/ecosystem_v1 --limit 64
python tools/data/compute_target_statistics.py \
  --config configs/hydrostate_ecosystem.py \
  --output /mnt/d/Hydrostate/samples/merged_v1/target_stats_ecosystem_v1.json
```

Product definitions: [Dynamic World](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_DYNAMICWORLD_V1),
[LAI](https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MCD15A3H),
[PML](https://developers.google.com/earth-engine/datasets/catalog/projects_pml_evapotranspiration_PML_OUTPUT_PML_V22a),
[albedo](https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MCD43A3),
[Landsat LST](https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C02_T1_L2).

### Extend LST observations and train with explicit target times

Use the completed `ecosystem_v1` pack as the baseline. The LST extension first
preserves qualifying same-day labels, then tests other same-day L2SP candidates,
then the L2 products corresponding to **existing Landsat L1 inputs**, nearest to
the original sample date first. Each candidate is tested on the actual window
pixels. The first candidate with usable pixels is selected; later candidates
are not mosaicked into it. Off-date labels also require valid pixels in their
paired Landsat input. Cloud masks and the **≤2 K** cutoff remain mandatory.

Each sample has at most one LST label map with its true UTC acquisition time.
An unavailable label stays masked. Base inputs, sample IDs and splits remain
unchanged. The result is a separate ordinary directory Zarr pack. Successful
downloads are resumable in `.lst_observation_v1.work/checkpoint.sqlite3`;
network failures are reported instead of becoming missing labels.

```bash
conda activate hydrostate

# Representative pilot; the later full run reuses this checkpoint.
OUTPUT_DIR=/mnt/d/Hydrostate/samples/merged_v1/labels/lst_observation_pilot_v1 \
  bash scripts/run_gee_lst.sh --limit 256
python tools/data/validate_gee_lst.py \
  --manifest /mnt/d/Hydrostate/samples/merged_v1/samples.parquet \
  --base-pack /mnt/d/Hydrostate/samples/merged_v1/labels/ecosystem_v1 \
  --pack /mnt/d/Hydrostate/samples/merged_v1/labels/lst_observation_pilot_v1 --limit 0

# Full extension; published destinations are immutable and cannot be overwritten.
bash scripts/run_gee_lst.sh
python tools/data/validate_gee_lst.py \
  --manifest /mnt/d/Hydrostate/samples/merged_v1/samples.parquet \
  --base-pack /mnt/d/Hydrostate/samples/merged_v1/labels/ecosystem_v1 \
  --pack /mnt/d/Hydrostate/samples/merged_v1/labels/lst_observation_v1 --limit 256

# Statistics use only the train split and the exact new label/time contract.
python tools/data/compute_target_statistics.py \
  --config configs/hydrostate_ecosystem_temporal.py \
  --output /mnt/d/Hydrostate/samples/merged_v1/target_stats_ecosystem_temporal_v1.json
HYDROSTATE_MODEL_PATH=/home/xiaoz/HydroState/pretrained/OlmoEarth-v1_2-Base \
  hydrostate-train configs/hydrostate_ecosystem_temporal.py
```

Use `SAMPLE_ROOT`, `BASE_PACK`, `OUTPUT_DIR`, `WORK_DIR`, `WORKERS`, `PYTHON` and
`GEE_PROJECT` to relocate collection. Training supports `HYDROSTATE_DATA_ROOT`,
`HYDROSTATE_LABEL_PACK`, `HYDROSTATE_LST_PACK`, `HYDROSTATE_TARGET_STATS`, and
`HYDROSTATE_MODEL_PATH`. A pilot pack covers only its selected sample IDs; use
the full pack for production statistics and training. Once the pilot is already
published, run validation or the full collection command directly.
The validator checks scene identity, actual acquisition time, input support and
2 K QA for all rows selected by `--limit`. Its all-task supervision/backward
check defaults to 64 of those rows (`--training-limit 0` checks all). This latter
check reads real masks/times and targets; it uses dummy predictions and does not
read satellite pixel values or measure model accuracy.

The 2026-10-16 representative 256-sample pilot retained seven baseline LST
samples and added 54 paired-input observations, for 61 usable samples and
72,245 valid 30 m label pixels under the unchanged 2 K cutoff. These are pilot
coverage results, not full-library counts or evidence of improved model accuracy.

All tasks use the same configurable `TargetTimeQuery` module
(`model.query_time`). The temporal configuration queries DW at its acquisition time, SMAP at its
observation time, ET/GPP on their eight-day interval, LAI on its four-day
interval, albedo on its 16-day retrieval support, and LST at its selected
overpass time. The pretrained encoder runs once; a task/time-conditioned pool
and the shared decoder run per task group (all nine cover classes share one
group). Exact acquisition slots replace the old 15-day cross-sensor merge.
This increases activation memory and decoder work; size batches for the training
GPU and retain gradient checkpointing. The existing configuration remains a
baseline for controlled comparisons.

This is retrospective retrieval/reconstruction, with all supplied images
available as context. Training a causal forecast would additionally require
removing later images before encoding. Changes in coverage do not establish
changes in predictive accuracy; evaluate the trained temporal model separately.

### Sampling environment

Use the project's Python 3.12 environment:

```bash
pip install -e ".[geo,gee,dev]"
earthengine authenticate
export GEE_PROJECT=your-registered-project
export OUTPUT_DIR=/mnt/d/Hydrostate/samples/centered8d_seed43_n10000
export SAMPLING_CONFIG=configs/data/sampling/centered8d.json
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

Default period: 2019–2024. Each input is 120×120 at 10 m. New runs default to
`configs/data/sampling/centered8d.json`: four half-open UTC slots
`[t−16,t−8), [t−8,t), [t,t+8), [t+8,t+16)`, where t is the DW target UTC date.
At least two distinct slots must contain a qualified sensor observation. Empty
slots are allowed; two sensors in one slot still count as only one active slot.
Each sensor contributes at most one scene per slot; all three need not be present.
There is no additional nearest/recent-observation threshold. Selection retains
the 50% window-valid fraction, existing QA masks, and the nearest qualified scene
to t within each slot. S1 uses the orbit of the nearest qualified S1 scene over
the full 32-day support, then selects only that orbit in every slot.
Training and regional preparation share the slot/orbit selector and acceptance
function. Centered sample selection evaluates QA on the same window-aligned 30 m
centres as regional masks, counting uncovered area as invalid. Equal-distance
candidates prefer the newer acquisition, then the lexicographically first asset.

Each new manifest row records `temporal_sampling`, `input_temporal_config`,
`slot_intervals`, and `observation_times_utc` (JSON fields for the latter three).
The original `inputs.<sensor_slot>.system:time_start` retains UTC milliseconds.
Zarr `observation_time` retains UTC epoch seconds with shape `(N,4,3)` in
S1/S2/Landsat order. Missing sensors have zero times and false validity, rather
than a copied observation. The encoder and time-query model read actual times,
including observations after t; slot centres are never substituted for them.

The original 20,000 samples and existing model experiments remain historical.
To resume an old collection, explicitly use `configs/data/sampling/past16d.json`:
four 16-day slots, at least two active slots, and a three-day recency limit.
Changing a preset requires a new output directory. Merging rejects different
temporal contracts, including duplicate physical windows; re-ingestion cannot
turn historical images into centered observations. New centered inputs must be
selected and collected again, and L2/label packs must reference the new manifest.
This temporal change does not alter satellite products: base collection still
uses Landsat L1; an L2 training experiment requires its same-acquisition L2 pack.

## Reuse a shared candidate pool

`scripts/run_gee_sampling.sh` defaults to `/mnt/d/Hydrostate/candidates` for the
shared pool and `/mnt/d/Hydrostate/samples/centered8d_seed${SEED}_n${SAMPLES}` for each new run.
The defaults are 10,000 samples and selection seed 43, creating a separate dataset
from the original seed-42 run in `/mnt/d/Hydrostate/samples`. Explicitly set
`OUTPUT_DIR=/mnt/d/Hydrostate/samples SEED=42
SAMPLING_CONFIG=configs/data/sampling/past16d.json` to resume that original run.
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

The historical dataset remains at `/mnt/d/Hydrostate/samples` unchanged.
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
  at the original window center. This is window-scale weak supervision,
  not full native SMAP support. No windows are linked.

An output pixel is observed when all its nine 10 m subpixels have at least one
valid sensor observation across the four slots. Water also requires this mask.
The current loss uses frozen training-only target means and population standard
deviations, calculated on these exact observed supports. Both physical predictions
and targets are standardized before Gaussian NLL. Each task learns one scalar
residual variance; the three task means are summed without manual task weights.
Missing observations contribute neither residual nor variance terms. Optional
station observations share their task's normalization and variance, with completely
observed footprints required. They enter the same task mean without a site multiplier.
Evaluation uses physical units; `hydro_score` averages RMSE divided by frozen training
std, independent of learned variances. See [HPC fine-tuning](HPC_FINETUNING.md)
for the formula, statistics generation, and the two-L40S execution path.

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

### Joint ecosystem training on HPC

Use `configs/hydrostate_ecosystem_temporal_hpc.py` for the 20,000 merged samples:
nine land-cover fractions, soil moisture, ET, LAI, GPP, black-sky and white-sky
albedo, and LST. All tasks use the shared target-time query module. LST uses
`labels/lst_observation_v1` with the unchanged 2 K uncertainty cutoff; the other
extension labels use `labels/ecosystem_v1`. Both packs remain ordinary directory
Zarr. Missing labels are masked, so a sample need not have all 16 labels.

Keep code in `/home/xiaoz/HydroState` and synchronize its committed `main` branch
through GitHub. Reuse `/scr/u/xiaoz/HydroState/samples_merged_v1`, including its
`manifests/merged_v1_n20000/samples.parquet`; place both packs under its `labels/`.
Before reusing an existing HPC manifest, verify that sample IDs, splits, grids and
target times match the local manifest. Storage path differences are allowed.

```bash
cd /home/xiaoz/HydroState
git pull --ff-only origin main
bash scripts/slurm/27_submit_ecosystem_training.sh
```

The launcher submits training-only statistics on CPU, followed by a GPU job with
an `afterok` dependency. It uses a separate immutable statistics file,
`target_stats_ecosystem_temporal_v1.json`, and validates its task/pack fingerprint.
Training requests one 48 GB GPU from `l40s`, BF16, batch 2,
gradient accumulation 8 (effective batch 16), 50 epochs and a 96-hour limit.
Logs, submission records and `work_dirs/ecosystem_temporal_JOBID` are under
`/scr/u/xiaoz/HydroState`. Each job verifies the committed source checksums before starting;
keep that checkout unchanged while jobs are queued or running. The submission
record prevents accidental duplicate submissions for the same commit.
Checksums are generated on the login node, so compute nodes do not need Git.
The cluster currently rejects comma-separated partition lists at submission;
select one partition per submission.

For the multilevel model with temporal queries and independent task adapters,
use `configs/hydrostate_ecosystem_multilevel_temporal_hpc.py` and submit another
training job after the ecosystem statistics have been generated:

```bash
cd /home/xiaoz/HydroState
git pull --ff-only origin main
bash scripts/slurm/28_submit_multilevel_temporal.sh
```

This reuses the same samples, packs and frozen statistics, without cancelling
existing jobs. It reads encoder blocks 4/8/12 in one pass, queries each depth at
the requested time, and uses the original multilevel fusion decoder, 16
independent output adapters and sigmoid task interactions. Interaction source
features are evaluated at the recipient's query time. The nine cover classes
share a query and are constrained to sum to one. WF/SM mask coupling stays off
because their output dates can differ; supervision matches the existing
16-variable temporal run. The optimizer follows the original multilevel model's
official layer-decay recipe and relative cosine floor, with batch 2 and
accumulation 8. Outputs go to
`/scr/u/xiaoz/HydroState/work_dirs/ecosystem_multilevel_temporal_JOBID`.

### Original GEE export transfers

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
