"""New, self-contained GEE library; independent of the historical merged_v1 data."""

output_dir = "/mnt/d/Hydrostate/samples/centered8d_16vars_v1"
project = "cropland-477906"
sampling_config = "configs/data/sampling/centered8d.json"
input_profile = "olmoearth_l1_v1"

samples = 10000
start_year = 2019
end_year = 2024
seed = 43
split_seed = 42
candidate_seed = 42
candidate_dir = "/mnt/d/Hydrostate/candidates"
candidates_per_class = 16

# Small windows use computePixels; no Google Drive staging is needed.
concurrency = 2
label_workers = 4
# Pixel-level audits; pack identity and manifest coverage are checked in full.
validation_limit = 64
