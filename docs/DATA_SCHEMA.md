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

Station supervision uses fixed arrays `site_value`, `site_valid`, and
`site_footprint`. Task order is water fraction, soil moisture, ET.

Input shards are immutable. A future label family is written under a new label
directory and joined to the existing inputs by `sample_id`.

