# Data preparation workflow

## 1. GEE-first sampling strategy

Google Earth Engine (GEE) is the primary source for satellite imagery and
large-area product labels. Select and export the required sample windows in
GEE; local processing validates the exports, applies model normalization, and
packages training-ready Zarr shards. The study domain is defined by the research
region and GEE data availability, rather than the intersection of locally stored
satellite scenes.

The planned training sequence is product-supervised training followed by
fine-tuning with station observations. This follows the broad strategy of
[Yao et al. (2026), BERTH](https://doi.org/10.1126/sciadv.aef3610): their methods
(pp. 7–8) describe nearest-neighbour resampling of water-cycle product labels to
500 m for supervised pretraining, followed by station-based fine-tuning with
500 m and 30 m inputs. HydroState's proposed use of nearest-neighbour product
labels on a 10 m output grid is an extension of that method, not an exact
reproduction. BERTH's pretraining soil moisture and ET products were AMSR-E/2
MCCA and ETMonitor; HydroState plans to use SMAP and PML instead.

This document specifies the agreed workflow. GEE sampling/export and conversion
into the current training format are not implemented yet. See section 7 for the
boundary between existing tools and planned work.

## 2. Freeze the sample locations, times, and splits

Each sample represents a 10 m, 128 x 128 pixel window with a stable `sample_id`.
Build one versioned `sample_index_v1.parquet` before exporting individual data
sources. It records window geometry, CRS, affine transform, target time or time
interval, sampling stratum, and dataset split.

- Stratify locations by land cover and water occurrence frequency, retaining
  representative land/water conditions and emphasizing intermittent-water and
  land-water transition regions. Water occurrence frequency is a sampling
  attribute, distinct from a single-scene Dynamic World probability label.
- Spread target dates across the selected years and seasons. The study region,
  date range, sampling counts, temporal windows, and quality thresholds must be
  explicit in the sampling configuration and frozen with each dataset version.
- Establish spatial train/validation/test blocks before generating samples.
  Keep all dates for the same location and all samples sharing a source SMAP
  grid cell in the same split; reconcile block boundaries with those cell IDs.
  Apply the same separation to station-centred samples used later.
- Limit repeated sampling within a coarse product cell, or assign explicit
  sampling/loss weights, so a small number of cells do not dominate training.
- Match availability per sample and task. Do not require every product to be
  available on every date. A missing label disables only that task's loss;
  exclude product-training samples that have no usable supervision at all.

The GEE domain geometry and availability metadata replace the local
`catalog/coverage_v1/common_coverage.geojson` as the primary sampling definition.
Local coverage inventories remain optional diagnostics.

## 3. Collect multisensor image inputs in GEE

For every sample, select observations around its target time according to the
configured temporal windows and organize them into the four temporal slots
expected by the current model. Preserve actual acquisition times for every
sensor; the sensors need not observe simultaneously.

| Input | GEE source | Preparation |
| --- | --- | --- |
| Sentinel-1 | `COPERNICUS/S1_GRD` | Select IW observations with VV and VH; preserve orbit metadata and quality masks. Values are already in dB. |
| Sentinel-2 | `COPERNICUS/S2_SR_HARMONIZED` | Select the encoder's 12 spectral bands; mask clouds, shadows, and invalid pixels. |
| Landsat | Collection 2 Tier 1 L1, such as `LANDSAT/LC08/C02/T1` | Select the encoder's canonical bands; verify radiometric representation and scaling against the pinned OlmoEarth preprocessing. |

Align input imagery to the sample CRS and exact 10 m grid, using documented
resampling choices for imagery and masks. Resampling does not change the native
information resolution of the source bands. Record band ordering, units,
scaling, processing versions, scene IDs, and quality decisions. GEE band names
must be mapped explicitly to the canonical order in [DATA_SCHEMA.md](DATA_SCHEMA.md).
Do not substitute Landsat L2 SR/ST bands for the expected L1 representation.

Export real observations with pixel-validity masks, `sensor_valid`, and
`observation_time`. Zero-fill missing modality/time slots and mark them invalid;
never duplicate another observation to fill a gap. Every usable sample must
have at least one valid satellite observation. Apply the pinned OlmoEarth
normalization exactly once before writing model-ready inputs.

## 4. Match product labels with nearest-neighbour sampling

| Product | Supervision | Spatial and temporal treatment |
| --- | --- | --- |
| Dynamic World (`GOOGLE/DYNAMICWORLD/V1`) | Water-class probability as a soft label | Align the 10 m probability and mask to the sample grid; retain the source observation date. |
| SMAP enhanced L3 soil moisture | Surface soil moisture | Nearest-neighbour mapping from the source grid; preserve AM/PM overpass information and retrieval quality flags. |
| PML ET | Evapotranspiration | Nearest-neighbour mapping from the source grid; preserve the product's composite interval, units, and scale factor. |

Pin exact collection IDs and versions for the selected years. SMAP collection
coverage and PML editions vary over time; changes must be explicit in metadata.
For example, PML V2.2a MODIS provides 500 m, 8-day mean daily ET with an `ET`
band scaled by 0.01 to mm/day. Match supervision to that interval rather than
treating it as an instantaneous observation or an unscaled interval total.

For the first baseline, map coarse labels directly to the 10 m sample grid using
nearest-neighbour sampling and evaluate the existing masked regression loss on
the resulting maps. Cross-patch coarse-footprint aggregation is not a prerequisite
for this baseline. Quality filtering and validity masks must survive export;
invalid or missing labels must never become valid zeros.

These are weak product labels: neighbouring output pixels can inherit the same
coarse source value. This preserves source values but creates no new fine-scale
observations and can favour spatially smooth predictions. Retain source-cell
IDs and native grid information to support balanced sampling, grouped evaluation,
and later comparisons with spatially aggregated losses without redownloading
imagery. The existing loss's shape-based pooling will not restore native support
once the label has already been resampled to the prediction grid.

Dynamic World's `water` band estimates the probability of complete water
coverage, not a measured subpixel water fraction. Keep that distinction in label
metadata and interpretation of the current `water` task. Temporal water
occurrence frequency and area fractions computed from classifications are
separate quantities.

Product agreement measures reproduction of the source products. Independent
station evaluation is needed to assess improvements beyond that agreement;
10 m output spacing alone does not demonstrate 10 m retrieval accuracy.

## 5. Export, validate, and package samples

Export only selected windows or manageable batches covering those windows.
Organize source exports independently, with restartable jobs keyed by
`sample_id`; avoid repeatedly downloading full regional archives.

Preserve the following information alongside arrays:

- Sample geometry, CRS, affine transform, split, target time, and label intervals.
- Scene IDs, acquisition times, modality/pixel masks, band order, and preprocessing.
- Label source and version, source coarse-cell IDs and native grids, units,
  scaling, temporal support, and quality flags.
- Sampling strata and any sampling or loss weights.

A local conversion step validates spatial alignment, dimensions, timestamps,
units, masks, and normalization, then writes immutable Zarr shards and the
training manifest `samples.parquet`. Keep source imagery and labels separately
archived so new label families or versions can reuse the imagery.

Proposed organization:

```text
/fossfs/xiaozhen/HydroState/
  catalog/gee_v1/
  manifests/sample_index_v1.parquet
  manifests/samples.parquet
  exports/gee/{s1,s2,landsat,dynamic_world,smap_sm,pml_et}/v1/
  stations/{ismn,fluxnet}/
  zarr/inputs/v1/
  zarr/labels/v1/
  logs/
```

The current reader expects each input shard to contain `s1`, `s2`, and `landsat`
together. Source-specific exports therefore need a packing step into this
combined input schema. Labels are stored separately and linked through the
manifest's `input_shard`/`input_row` and `label_shard`/`label_row` references.
The detailed export metadata can remain in sidecar tables keyed by `sample_id`.

## 6. Station observations and subsequent fine-tuning

Product supervision is followed by station-based refinement using ISMN soil
moisture and FLUXNET ET observations. Keep training, validation, and test stations
separate, consistent with the spatial split established above. Match station
depth, units, observation intervals, and spatial footprint to each target.
Independent test stations must not be used for refinement.

### ET: FLUXNET

The existing tools discover and spatially filter sites, download observations
through FLUXNET Shuttle, and convert half-hourly/hourly latent heat to interval
ET using `ET = LE * seconds / 2.45e6`. They retain raw LE, its source field,
interval, and quality field. Temporal aggregation to the selected ET target
interval is a subsequent preparation step.

Install Shuttle with:

```bash
pip install "git+https://github.com/fluxnet/shuttle.git"
```

The existing job is `scripts/slurm/02_prepare_fluxnet.sh`. It currently hardcodes
the local common-coverage geometry. Update that path to the frozen GEE study
geometry before using it for this workflow. Its two Python tools accept different
arguments, so do not pass `--eligible-area` through the job's shared trailing
arguments to the ET normalization command.

### Surface soil moisture: ISMN

Download an ISMN archive from the portal, then use
`tools/data/stations/import_ismn.py` with the frozen study geometry and the
required depth range. The existing `scripts/slurm/03_import_ismn.sh` job uses
`ISMN_ARCHIVE`, defaults to 0–10 cm, and still references the local common-coverage
geometry. Align the selected station depths with the intended surface-soil
moisture target before fine-tuning. Outputs are station metadata and normalized
Parquet observations in m3/m3.

```bash
pip install -e ".[geo,stations]"
```

Station import retains all available years unless `--start` and `--end` are
specified. Availability and quality are subsequently matched to sample targets.

## 7. Implementation boundary

Existing components include the local coverage scanner, station ingestion and
ET conversion tools, combined-input Zarr writer/reader, manifest sampler, masked
product and station-footprint losses, and MMEngine training framework.

The following work is still required for this plan:

- GEE domain/sample-index generation, scene and label matching, export jobs,
  and restart tracking.
- Export validation and packing into the current input/label Zarr schema,
  including provenance, source-cell IDs, and split enforcement.
- Coarse-cell sampling controls and grouped product/station evaluation.
- Station-job geometry updates and target-specific time/depth alignment.
- Separate product-training and station-fine-tuning configurations. The current
  model can combine product and station losses in one run; it does not yet
  implement the planned staged training workflow.

[DATA_SCHEMA.md](DATA_SCHEMA.md) still describes coarse-label aggregation and
[IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) records earlier data-source
assumptions. Those contracts need reconciliation when implementing this plan;
the nearest-neighbour weak-supervision policy here is the agreed baseline.
