"""Frozen station production: existing joint draw, then same-frequency replacements."""

project = "cropland-477906"
min_input_fraction = 0.5
min_active_slots = 2
min_date_gap_days = 30
seed = 43
homogeneous_fraction = 0.95
spatial_valid_fraction = 0.95
class_confidence = 0.60
footprint_min_mass = 0.95
min_free_gib = 20
footprint_root = "/fossfs/xiaozhen/HydroState/site/footprints_icos_20261103"
split_root = "/fossfs/xiaozhen/HydroState/site/splits_20261103"
product_manifest = "/scr/u/xiaoz/HydroState/samples/retrieval_eight_20000_06aba36/samples.parquet"
