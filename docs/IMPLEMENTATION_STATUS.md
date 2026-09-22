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


## GEE pipeline update

The current pipeline supersedes the local-ingestion plan above: all imagery comes
from GEE, including Landsat L1. Global climate/month/occurrence sampling and
verified Drive transfer are implemented in `hydrostate/data/gee_sampling.py`.
Stored samples are 120×120 at 10 m; training outputs are 40×40 at 30 m.
Authenticated GEE/Drive pilot validation passed as detailed below; full global
sampling quality and production-scale throughput remain unvalidated. See DATA_PREPARATION.md.

## Real-data pilot verification (2026-09-14)

Three real GEE samples (two Poyang Lake, one Arkansas) were exported to Drive,
downloaded with MD5 checks, ingested into Zarr and removed from their dedicated
Drive folders. Read-only audit found zero remaining files in both pilot folders.
The third sample has valid water, SMAP and PML supervision.

Original OlmoEarth v1.2 Base weights passed GPU forward-only evaluation on all
three samples: four input slots, 120×120 pixels, three finite 40×40 outputs.
RTX 4060 BF16 forward time was 0.19–0.66 s/sample, peak allocated memory 917 MiB.
This is an execution check, not retrieval-accuracy validation; task heads are new.
Artifacts: outputs/pilot_transfer_audit.json and outputs/gpu_forward_check.json.

Pilot fixes include explicit EPSG:6933 WKT, full source asset paths, request/task
ID reconciliation and exact-grid cropping of GEE rounding borders. The adapter
uses an equivalent broadcast attention mask and optional gradient checkpointing.
15 local tests passed, including attention output/gradient equivalence.


## Multiscale supervision update (2026-09-14)

- Water: 10 m binary DW aggregated to 30 m fraction with complete label masks.
- ET: native-grid polygon overlaps in local equal-area coordinates; only complete
  cells are supervised. GEE reports EPSG:4326 with 1/240-degree spacing for the
  pilot PML assets, so nominal 500 m integer pooling would be incorrect.
- SMAP: observed-window prediction mean against the original center label,
  with reduced product loss weight. This remains weak supervision, not full SMAP
  footprint consistency. No cross-window association.
- Training and evaluation share the same aggregation and input-coverage masks.
  Losses and the combined metric use configurable physical reference scales.
- Schema v3 stores variable native-cell weights, padded by `hydro_collate`;
  spatial flips transform weights too. Legacy ET labels without weights fail
  explicitly. `--reingest-from` rebuilds shards from MD5-verified local TIFFs.

The three existing real samples were converted to
`outputs/gee_multiscale_pilot/samples.parquet`. They contain 4, 2 and 4 complete
valid PML cells respectively; all ten survive the input-coverage check. The Arkansas
sample also supplies one valid SMAP window constraint. All three have 1600 valid
water output pixels. Original encoder weights passed RTX 4060 BF16 forward/loss
and metric evaluation, including a batch of two different native-cell counts.
Peak allocated GPU memory was approximately 1091 MiB. No optimizer was created
and no model parameters were updated. Report: `outputs/multiscale_forward_check.json`.
These checks establish execution, not retrieval accuracy of the newly initialized heads.

20 tests pass, including area overlap at nonaligned boundaries, exclusion of partial
cells and unobserved support, augmentation/collation, and singleton R² handling.
MMEngine's configured DataLoader also passed on the converted real samples.


## Seven frequency bins and block screening (2026-09-15)

The sampler now includes frequency 0 and 100 as separate strata alongside the
five intermediate bins. Seven bins receive equal quotas; year-month balancing
and the half-equal/half-area climate allocation are retained. YearlyHistory
non-no-data observations distinguish observed dry land from missing coverage.
Native-resolution climate screening skips empty ocean blocks before expensive
30 m sampling and histograms, with conservative 2 km edge buffering. The sampling
strategy identifier prevents mixing different plans. No five-bin cache migration
is implemented. Existing production results were removed at the user's request;
the seven-bin production run has not been started.


## Shared candidate pool (2026-09-15)

Candidate blocks now use an independently configured shared directory and spatial
seed. Per-run sample counts, dates, selection seeds and transfer state are separate.
Pool metadata guards against incompatible reuse; locks prevent racing builders.
Existing production directories were not moved: `candidates` and `samples_10000`
are links into the currently running `samples` directory. No new sampling process
was started by this update.
