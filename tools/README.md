# Command-line tools

Current entries are thin wrappers around `hydrostate` package services:

- `data/collect_gee_dataset.py`: collect or preview current data.
- `validate_zarr.py`: audit completed data (or explicit partial pilots).
- `data/compute_target_statistics.py`: current training-only moments.
- `data/prepare_training.py`: audit/prepare or verify a paired training release.
- `train.py`, `test.py`: MMEngine training and evaluation.
- `watch_mtlora_collection.py`: explicitly launched HPC submission watcher.

`data/audit_cover_thresholds.py` is a diagnostic experiment, not a collection entry.
`data/stations/` and archive transfer utilities remain available for separate work.
`figures/`, `validation/`, `experiments/` contain analysis/experiment tools;
`legacy/` contains historical data preparation, regional export and model experiment CLIs.
The old regional export tools do not support the current seven-head data contract.
