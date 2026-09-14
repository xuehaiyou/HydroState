# HydroState sampled-data contract

## Coordinate and sample convention

Each row in `samples.parquet` identifies one 10 m, 128 x 128 pixel window and
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

- `s1`: `(N,4,2,128,128)`, VV then VH in dB.
- `s2`: `(N,4,12,128,128)`, canonical OlmoEarth S2 order.
- `landsat`: `(N,4,11,128,128)`, canonical OlmoEarth Landsat order.
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

## Label shard

The standard label store contains 10 m maps and validity masks for `water`,
`soil_moisture`, and `et`. A coarse product must be compared after aggregating
predictions to its support; it must not be presented as independent 10 m truth.

Station supervision uses fixed arrays `site_value`, `site_valid`, and
`site_footprint`. Task order is water fraction, soil moisture, ET.

Input shards are immutable. A future label family is written under a new label
directory and joined to the existing inputs by `sample_id`.

