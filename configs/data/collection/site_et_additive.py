"""EBR-qualified existing sites; 1,200 successful new windows, fixed revised splits."""

_base_ = "./site_supplement.py"
split_root = "/fossfs/xiaozhen/HydroState/site/et_revision_20261110/splits"
sampling_reference = "/fossfs/xiaozhen/HydroState/site/et_revision_20261110/inventory.parquet"
success_targets_by_split = dict(train=828, validation=190, test=182)
