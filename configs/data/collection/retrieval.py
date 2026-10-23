"""Current product-supervised collection; fixed contract, new output directory."""

output_dir = "/mnt/d/Hydrostate/samples/retrieval_5x8_native_v2"
project = "cropland-477906"
samples = 20000
start_year = 2017
end_year = 2024
seed = 43
split_seed = 42
candidate_seed = 42
candidates_per_bin = 24
candidate_block_degrees = 2
candidate_block_batch = 16
candidate_pool_factor = 3
candidate_dir = "/mnt/d/Hydrostate/candidates/retrieval_5x8_v1"
climate_asset = "users/xiaozhen6666666/cloud/Beck_KG_V1_present_0p0083"
max_dates_per_location = 4
min_date_gap_days = 40
split_margin_m = 35000
# Bounded in-flight samples; IO launcher still uses only two CPU cores.
concurrency = 4
min_active_slots = 2
min_input_fraction = 0.5
water_confidence = 0.35
ice_confidence = 0.60
land_confidence = 0.85
land_max_water_ice_probability = 0.10
min_known_fraction = 0.95
max_water_ice_fraction = 0.05
min_tile_land_fraction = 0.95
min_cover_pixels = 1
