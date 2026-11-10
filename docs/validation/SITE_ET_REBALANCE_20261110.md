# ET split revision and same-station date supplement

User-approved first increment: 1,200 successful ET windows. Baseline uses the
98 physical stations passing the full-day, measured EBR protocol documented in
[SITE_ET_EBR_PROTOCOL_20261110.md](SITE_ET_EBR_PROTOCOL_20261110.md).

The baseline 1,194 ET windows originally split 428/332/434. Whole spatial groups
are retained or promoted from evaluation to training; existing training groups
never enter evaluation. SM assignments and groups shared with SM remain fixed.
Within frequency bins, optimize window totals, station totals, then land cover,
climate, year and month balance. Mixed-frequency spatial groups remain fixed.
Result: 848/169/177 windows and 69/14/15 physical stations (train/validation/test).

Supplement success targets are 828/190/182; the resulting target is 1,676/359/359
windows (approximately 70/15/15). Draw from existing quality-qualified dates of
these same stations, within the 2015–2025 energy audit scope and available inputs.
Exclude previously produced physical-station/time-window pairs and deduplicate
new pairs. No 30-day gap. Water frequency is the first sampling priority, then
land cover × climate, year, month and station diversity. Existing accepted windows
seed the counters. Failed candidates are replaced within their fixed split.
Exhausted frequency pools are reported; no quality relaxation or new stations.

The (0,20)% frequency bin contains only two independent spatial groups, so a
three-way split of this bin is impossible without station leakage. Additional
dates increase temporal coverage, not independent station count. Previously
examined evaluation results remain development evidence, not pristine blind tests.

Reuse uncorrected ET labels and existing source/time quality rules, known footprint
aggregation, otherwise 95% homogeneous and valid support with class confidence
0.50. Preserve conservative product overlap exclusions. Reuse old Zarr roots via
new manifests; never rewrite published shards. New dates form a separate cohort.
Training statistics must be recomputed for the revised manifests before activation.
Preparation is not publication or training activation.

## Execution record

- Frozen HPC code: `89981c3`, `/home/xiaoz/releases/site_et_additive_89981c3`.
- Revision manifests and candidate plan:
  `/fossfs/xiaozhen/HydroState/site/et_revision_20261110`.
- Deduplicated unused pool: 23,030 windows (train 15,740; validation 3,506;
  test 3,784). Frequency bins: zero occurrence 22,801; (0,20)% 229.
- Launcher/status/log: `/fossfs/xiaozhen/HydroState/site/runs/et_additive_20261110/`.
- Additive Zarr cohort: `/fossfs/xiaozhen/HydroState/site/prepared/site_et_additive_20261110`.
- IO-node collection uses 4 workers, 2 CPU affinity, initial 12-attempt smoke
  collection and scientific audit before automatic continuation. No automatic
  publication or training activation. Read `status.json` for live state.
- Related local tests: 12 passed; Ruff passed. These are code checks, distinct
  from the HPC smoke audit recorded by the launcher.

## Empty smoke recovery

The first 12 candidates failed the unchanged spatial quality rules, leaving no
manifest for audit. The old launcher incorrectly invoked audit on this empty
result. `scripts/run_site_et_additive.py` now resumes existing attempts in batches
of at most 24, up to 240 total smoke attempts across restarts. It audits only a
nonempty sample set, then automatically continues production. Budget exhaustion
or candidate exhaustion are explicit states; genuine audit errors still stop the
run and retain stderr. Existing cache and rejections are reused without relinking.
The production contract, EBR list, split targets and quality thresholds are unchanged.
