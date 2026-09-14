# Implementation status

Implemented:

- MMEngine registries, Runner configurations, AMP, DDP-ready data loaders, and
  checkpoint selection.
- Official rslearn OlmoEarth v1.2 Base adapter with real observation times.
- Shared spatial decoder and joint water-fraction, soil-moisture, and ET heads.
- Masked product losses, station-footprint losses, and regression metrics.
- MMEngine LLRD optimizer constructor with Base's 12-layer, 0.65 decay recipe.
- Immutable sampled-Zarr writer, reader, manifest sampler, and validator.

Data ingestion is intentionally not guessed yet. The product and station paths,
variable names, units, resolutions, and time-support definitions have not been
provided. Those are required to generate valid sample centers and labels.

The available source layouts were checked read-only:

- S1 contains paired `Sigma0_VV_db.tif` and `Sigma0_VH_db.tif`, so it appears
  already expressed in dB. RTC/projection metadata still needs verification.
- S2 is organized as Level-2A `.SAFE` scenes.
- Landsat is Collection 2 Level-2 (`LC08_L2SP`) with SR/ST products. OlmoEarth's
  documented encoder input is Collection 2 Level-1 with canonical bands
  `B8,B1,B2,B3,B4,B5,B6,B7,B9,B10,B11`. The L2 store must not be silently
  presented as that Level-1 tensor. Before Zarr generation, choose either a
  matching Level-1 source or define and validate an explicit L2 adapter.

