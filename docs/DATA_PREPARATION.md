# Data preparation workflow

## 1. Build the immutable eligible area

The first CPU job inventories actual raster footprints for local S1, S2 and
Landsat L2. It reads one representative band per scene, transforms its bounds
to WGS84, writes three Parquet catalogs, and intersects sensor coverage across
all locally available years. It does not copy imagery. The output GeoJSON records
the observed date range of each sensor; these ranges do not guarantee simultaneous
observations at every location. Sample dates must subsequently be matched against
the scene catalogs at each location.

```bash
mkdir -p /fossfs/xiaozhen/HydroState/logs
sbatch scripts/slurm/01_build_local_coverage.sh
```

Dates are optional for all three CPU jobs. For example,
`sbatch scripts/slurm/01_build_local_coverage.sh --start 2021-01-01 --end 2024-12-31`
restricts coverage calculation while retaining all scenes in the catalogs.
Without these arguments there is no start or end cutoff. Station import similarly
retains all available years by default. Product availability and sample dates are
matched later, with missing labels represented by validity masks rather than a
global requirement that all label products overlap in time.

The authoritative result is
`catalog/coverage_v1/common_coverage.geojson`. Inspect it before GEE sampling.
The later GEE task must sample only inside this geometry, stratify intermittent
water frequency into bins strictly between 0 and 1, and add station-centred
anchor samples. It then freezes one `sample_index.parquet`; all source/label
jobs join by `sample_id` and write only their own Zarr group.

For a quick scanner test on the login node, append `--limit 10 --workers 2` to
the Python command. A limited run is diagnostic only and must not be used as
the sampling domain.

## 2. Station observations

### ET: FLUXNET

Install the official FLUXNET Shuttle once:

```bash
pip install "git+https://github.com/fluxnet/shuttle.git"
```

Then submit:

```bash
sbatch scripts/slurm/02_prepare_fluxnet.sh
```

The job obtains a current site snapshot, spatially filters it (and applies dates
only when explicitly supplied),
downloads selected sites through Shuttle, and converts half-hourly/hourly
latent heat to interval ET using `ET = LE * seconds / 2.45e6`. Raw LE, its
source field, interval and quality field are retained. Provider authentication
and license acceptance remain the user's responsibility.

### Surface soil moisture: ISMN

ISMN requires free portal registration and acceptance of its terms. In the
portal, request soil moisture covering the project period and download the
resulting archive. The script applies exact common-coverage and 0–10 cm filters.

```bash
pip install -e ".[geo,stations]"
ISMN_ARCHIVE=/path/to/download.zip sbatch scripts/slurm/03_import_ismn.sh
```

Outputs are normalized Parquet tables: station metadata and observations in
m3/m3. Site data serve as calibration/independent-validation anchors; GLASS SM
and PML ET remain wall-to-wall product supervision.

## 3. Storage boundary

```text
/fossfs/xiaozhen/HydroState/
  catalog/coverage_v1/
  manifests/sample_index_v1.parquet
  stations/{ismn,fluxnet}/
  zarr/{s1,s2,landsat,dynamic_world,glass_sm,pml_et,station}/v1/
  logs/
```

Each producer is restartable and independent. No Zarr store contains unrelated
source imagery. Adding a variable creates a sibling store rather than mutating
an existing store. The immutable manifest is the relational key.
