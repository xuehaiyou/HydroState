# HydroState sampled-data contract

## Coordinate and sample convention

Each row in `samples.parquet` identifies one 10 m, 120 x 120 pixel window and
four temporal slots. Spatial splitting is completed before shards are written;
all times for the same site and spatial tile belong to one split.

Required manifest columns are:

| Column | Meaning |
| --- | --- |
| `sample_id` | Globally unique stable identifier |
| `split` | `train`, `validation`, or `test` |
| `input_shard` / `input_row` | Satellite shard and row |
| `label_shard` / `label_row` | Hydro-label shard and row |
| `sample_weight` | Optional positive sampling weight |

Recommended metadata columns are `tile_id`, `site_id`, `target_time`, and
`label_source`.

## Input shard

Arrays use `(N,T,C,H,W)` order:

- `s1`: `(N,4,2,120,120)`, VV then VH in dB.
- `s2`: `(N,4,12,120,120)`, canonical OlmoEarth S2 order.
- `landsat`: `(N,4,11,120,120)`, canonical OlmoEarth Landsat order.
- `sensor_valid`: `(N,4,3)`, modality order S1, S2, Landsat.
- `observation_time`: `(N,4,3)`, UTC Unix seconds in the same modality order.
- `{s1,s2,landsat}_valid`: pixel-validity masks.

The imagery arrays are model-ready and normalized using the exact
OlmoEarth preprocessing version stored in group attributes. A missing
modality/time is zero-filled and marked false in `sensor_valid`; it must never
be filled by duplicating another observation.

New centered sample runs declare `temporal_sampling=centered_8d_v1` and store
`input_temporal_config`, `slot_intervals`, and `observation_times_utc` as JSON
manifest columns. The four UTC intervals are `[t−16,t−8)`, `[t−8,t)`, `[t,t+8)`,
`[t+8,t+16)`, and at least two distinct slots require a qualified sensor.
Other slots may be empty; no recency gate is imposed. Timestamps belong to each
sensor observation, not to the slot centre;
the model-facing `(N,4,3)` epoch-second array is unchanged. Zarr attributes retain
the temporal contract. Unversioned old samples retain their historical four
16-day slots; temporal policies cannot be mixed by the dataset merger.

S2 canonical band order is:

`B02,B03,B04,B08,B05,B06,B07,B8A,B11,B12,B01,B09`.

Landsat canonical order is:

`B8,B1,B2,B3,B4,B5,B6,B7,B9,B10,B11`.

The current `/fossfs/DATAPOOL/Landsat_L2` inventory is Collection 2 Level-2,
whereas this canonical OlmoEarth input corresponds to Collection 2 Level-1.
The sample builder must not silently substitute SR/ST bands. It must use a
documented L2 adapter or a matching Level-1 source and record that decision in
the shard attributes.

## Training grid

Stored arrays are 120×120 at 10 m. The preprocessor aggregates binary DW water
in 3×3 blocks, requiring all nine masks valid. Model outputs are 40×40 at 30 m.
Water is supervised per output pixel. SMAP supervises the mean of observed output
pixels using the original window-center label (selected before augmentation).
PML supervises area-weighted output averages inside complete native product cells.
These three observational supports also apply to evaluation metrics.

## Label shard

The standard label store contains 10 m maps and validity masks for `water`,
`soil_moisture`, and `et`; coarse maps are nearest-neighbor storage, not 10 m truth.
Schema v3 additionally stores `et_native/{row}/weights` (G×40×40), `value` (G),
and `valid` (G). G counts fully covered, valid native PML cells. A single zero-weight,
invalid row represents no usable cell. Positive weights are normalized polygon
intersection areas in a local equal-area projection, using densified native and
output boundaries. Native CRS/affine are retained in the sample attributes.
Only cells entirely within the window are retained; no cross-window aggregation
is performed. At training time a cell is also excluded if any contributing output
pixel lacks input observations. `hydro_collate` pads G within each batch with invalid
zero-weight rows, and augmentation flips the weights with the inputs.

SMAP window-mean supervision remains a spatial approximation, not a full SMAP
footprint average. Legacy labels with valid ET but no native weights fail explicitly;
re-ingest verified GeoTIFFs into a new directory with `--reingest-from`.

The optional `hydrostate_staged_light_land_sm.py` configuration replaces SM's
unweighted window mean with a land-area-weighted mean using detached WF. By
default it uses valid observed WF labels in both training and evaluation; unknown
WF inside input-observed support excludes that window from SM supervision.
Prediction maps always use predicted WF for their NoData mask. This changes SM's
support without changing stored labels, and does not resolve the SMAP footprint
approximation. See `MODEL_ARCHITECTURES.md` for thresholds, optional soil-domain
mask, fixed normalization scales, and the prediction/validation distinction.

Station supervision uses fixed arrays `site_value`, `site_valid`, and
`site_footprint`. Task order is water fraction, soil moisture, ET.

Input shards are immutable. A future label family is written under a new label
directory and joined to the existing inputs by `sample_id`.

## Versioned input and label packs

`HydroStateZarrDataset` now accepts `input_packs`, `label_packs`, and `tasks`.
The original manifest columns and Zarr layout remain readable without any new
configuration. Existing satellite inputs remain the base input contract; packs
extend these inputs with auxiliary variables, or supply selected regression labels.

```text
dataset/
  samples.parquet
  zarr/...                       # Existing satellite inputs and hydro labels
  inputs/terrain_v1/
    metadata.json
    index.parquet
    _SUCCESS.json
    data/part-00000.zarr
  labels/temperature_v1/
    metadata.json
    index.parquet
    _SUCCESS.json
    data/part-00000.zarr
```

Each pack has one `kind` (`input` or `label`), source description, version, and a
variable specification containing shape, dtype, physical units and grid. A pack
may contain several related variables. Data are written into shards of up to 256
indexed samples by default. `chunk_samples` defaults to one; the GEE ecosystem
producer buffers 16 samples per ordinary Zarr chunk. Unused capacity in the last
shard has no index entries. Stores remain normal Zarr directories, not ZIP archives.

The index contains `sample_id`, `shard`, `row`, and `sample_fingerprint`. Shard paths
are relative to the pack. The fingerprint binds extensions to the existing ID,
split, location/grid and target time when those fields exist in the base manifest.
It deliberately excludes storage paths and selected input/label products. Never
recompute an existing sample ID when adding variables. Indices must be unique and
their IDs must belong to the supplied base manifest. A pack may cover only a subset
of that manifest; omitted samples receive zero values and false validity masks.

Only selected variables are read. A variable cannot come from two selected packs
of the same kind. Selecting a label pack replaces that variable's legacy label,
including missing rows: there is no silent fallback to the old source. Combining
multiple products for one target requires an explicit source-combination adapter.

For every variable `x`, the pack stores `x` and an equally shaped `x_valid` mask.
Values marked valid must be finite. Missing values are sanitized to zero at
publication. Supported pack shapes exclude the shard dimension:

| Kind | Shape | Metadata |
| --- | --- | --- |
| Static auxiliary input | `(C,H,W)` | `temporal="static", grid="input"` |
| Sequence auxiliary input | `(T,C,H,W)` | `temporal="sequence", grid="input"` |
| Pixel regression label | `(1,H,W)` | `grid="input"` or `"output"` |
| Window-mean regression label | `(1,1,1)` | `grid="scalar"` |
| Native-pixel regression label | `(capacity,)` | `grid="native", spatial_shape=[H,W]` |

Sequence inputs also store `x_time` with shape `(T,)`, UTC epoch seconds. Producers
must reproject to the exact sample input grid and select the intended temporal
support before publication; the importer does not infer georeferencing or time
alignment from array shape. Input-grid pixel labels are averaged to the output
grid with complete-mask requirements. Output-grid labels are already aligned.
Window-mean labels must represent the window observation being supervised, not an
arbitrary larger native pixel. Native packs additionally store `x_weights` with
shape `(capacity,H,W)`: nonnegative area weights summing to one per valid complete
source pixel. Invalid rows are zeroed; readers trim padding and `hydro_collate`
pads independently per task. New tasks declare `support="native", grid="native"`.
Existing PML geometry remains in legacy shards and retains `pml_native` support.
All native tasks require every contributing output pixel to have valid inputs.

Labels may additionally declare `time_support="instant"` or `"interval"` and
store `x_time` with shape `(2,)`, int64 UTC epoch seconds. Instantaneous labels
use `[time,time]`; intervals use `[start,end)` and describe product retrieval
support, which need not be a simple temporal average. Valid labels require a
positive start; absent labels may use `[0,0]`. Input-sequence times remain `(T,)`.
`labels/lst_observation_v1` uses one `(1,40,40)` LST map and one actual acquisition
time per sample, with uncertainty already screened at **≤2 K**. Its index records
the selected L2 asset, matching L1 asset/slot for off-date observations, and tested
candidate quality. It does not mix several dates into one temperature map.

With `target_time_queries=True`, the dataset adds
`inputs["target_time"][task]` of shape `(2,)` (batched `(B,2)`). Existing ecosystem
pack observation times are recovered from their frozen index provenance, and
SMAP/PML times from the base manifest. Missing labels use the original sample
date as an inference query while retaining false supervision masks. This option
is part of the frozen training contract, so it requires new target statistics.
Valid labels without observation times are rejected. A replacement label pack
must supply its own times; it never inherits the replaced product's timestamp.
To extend another variable, declare its `time_support`, store its `x_time`, and
add its task/head configuration. The shared query module requires no new branch.

`TargetTimeQuery` in `hydrostate/models/query_time.py` is the shared query-time
module for every task, with no product-specific branches. Configure it through
`model.query_time`. `TemporalHydroStateModel` keeps each satellite acquisition
separate, pools the encoder's temporal features using this module, and then applies
the shared spatial decoder and task heads. The nine cover fractions share one
query and one decoder pass. This variant disables the wrapper's 15-day timestamp
merge and preserves per-pixel input masks through the encoder. Prediction records
include each task's actual query bounds in `target_time`; the original sample
date in `metainfo` remains an identity field. At inference, callers supply the
desired per-task query bounds through the same input dictionary.

The temporal variant is retrospective retrieval/reconstruction: the encoder may
use all supplied images, including images after a particular task's query time.
It is not a causal forecasting model. New pooling parameters need training; old
task checkpoints are not equivalent to trained temporal-query checkpoints.

Labels may declare `uncertainty_units` and store an equally shaped
`x_uncertainty`. An optional `default_max_uncertainty` applies on reading;
explicit pack selection `max_uncertainty={"lst": 2.0}` also pins the threshold
in the frozen training data contract. Stored validity describes source quality;
training validity additionally applies this threshold. Ecosystem LST retains
clear-sky observations and uncertainty but excludes values above 2 K by default.
Changing a filter requires new training statistics, not new imagery downloads.
Optional per-sample provenance JSON in the index records source asset IDs,
observation timestamps, composite intervals and actual source projections.

`model.fraction_groups` applies a joint softmax to named identity/logit pixel
heads. The ecosystem configuration groups all nine exhaustive DW classes,
including existing water. Source class indicators share a mask and aggregate
from 10 m to 30 m; classification probabilities are not treated as area fractions.

Task specifications are centralized in `hydrostate/tasks.py`. Defaults preserve
WF pixel supervision, SMAP center/window-mean supervision and native PML supervision.
An explicit task mapping selects the active tasks. Each additional task requires
`support` and `grid`; new tasks have no legacy station slot unless explicitly
configured. The base, staged, and multilevel models, losses, statistics, metrics
and optional visualization hook use the selected tasks. Add the corresponding
regression head in the model configuration.

`AuxiliaryInputFusion` encodes selected auxiliary variables separately and adds
their features to the satellite encoder output(s). It requires fixed per-channel
mean/std, either declared physical scales or training-only statistics. Static
inputs use `temporal="static"`; sequences require the explicit `temporal="mean"`
masked time reduction. Timestamps are retained but are not consumed by this mean
adapter. Use a custom registered fusion module when ordering/time gaps matter.
An entirely missing auxiliary input contributes exactly zero. Satellite masks
continue to determine prediction coverage. Replacing satellite bands/modalities
requires their own pretrained-model/normalization adapter.

Publication uses a private partial directory and an atomic rename to a new version.
Writers refuse to replace an existing destination. The success marker checksums
metadata, which checksums the index; raw chunk corruption is not detected by those
metadata checksums. Treat published data as immutable and keep a backup. Updates
are new packs, not in-place edits. Version-2 target statistics include task support
definitions and selected pack metadata/index fingerprints; the training CLI rejects
statistics computed with a different selection. Original version-1 statistics and
checkpoints remain valid for the original configuration.

Copy complete pack directories when moving a dataset and preserve their paths
relative to `data_root`. The existing snapshot/merge tools still manage legacy
manifest shard columns; they do not automatically discover auxiliary pack folders.

