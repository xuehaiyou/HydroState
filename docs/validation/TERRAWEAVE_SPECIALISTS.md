# TerraWeave optical and radar specialist comparison

Three arms reuse the six-epoch manual Cover and station protocol, including its reference selection, label statistics, losses, model selection and complete test. The frozen optical encoder is Prithvi-EO-2.0-100M-TL (86,237,186 parameters). The radar encoder is SoftCon ViT-S/14 (21,553,920). Fusion has 107,791,106 frozen parameters. The two single-source models have different capacities.

HLS L30 and S30 provide blue, green, red, narrow NIR, SWIR1 and SWIR2 on the original 30 m output grid. Daily QA-masked reflectance is averaged across valid dates within each original slot. Only Fmask bits 0–3 are excluded. Each optical slot uses the published HLS normalization, midpoint date and window location. SAR uses the existing VV/VH time composites and the project's normalized radar values clipped to [0,1]. Both encoders run separately per slot, with an 8 by 8 patch grid and four exits after blocks 3, 6, 9 and 12. Regridding for TerraWeave leaves the observation footprint unchanged. Only active sources supply valid pixels and quality information.

The three arms are selected by the names prithvi, softcon and prithvi_softcon in hydrostate.experiments.dense_adaptation. SPECIALIST_WEIGHTS holds the released checkpoints and author implementations. SPECIALIST_HLS holds the supplementary files indexed by sample ID. Optical training depends on successful completion of hydrostate.data.retrieval.specialist_hls. Window lists stay fixed. Effective supervision counts can differ with sensor coverage and should accompany independent validation metrics.

## HKU submission, 27 November 2026

All three training jobs use commit 89b3bf8 on the L40S partition. SoftCon is job 4520288, Prithvi is 4520289, and fusion is 4520290. The run directories are under /scr/u/xiaoz/HydroState/experiments/terraweave_specialists. The six-epoch reference is foundation_fourstage_c6caa6d/d2moe_v2_dedicated/config.py. The fixed splits have 10,912, 3,681 and 11,168 windows.

All three models completed a CPU forward, loss and backward pass on a real selected training window. Two real HLS downloads passed the reflectance and coverage checks. Full HLS collection runs on hpc2021-io2 with two CPUs and four concurrent requests. The first Slurm CPU attempt (4520282) lacked outbound network access. The IO controller releases the two optical jobs after the input collection succeeds. Earth Engine reported restricted noncommercial quota. Initial submission provides no validation accuracy.

## Replacement using the published S2 samples

The current optical arms read Sentinel-2 L2A from the original Zarr, in the order B2, B3, B4, B8A, B11 and B12. The reader inverts the published per-band encoding to reflectance DN, aggregates complete 30 m cells on the existing aligned grid, and applies the frozen Prithvi checkpoint normalization in the encoder. Masked cells remain invalid. Time and location come from the original window. Both arms use the same S2 observations and original six-epoch protocol. The completed SoftCon control remains the SAR reference. The HLS download has been stopped and removed. Counts of valid observations can differ across sensors, so direct comparisons should also use common valid support.

HKU L40S accepted jobs 4520958 (S2) and 4520959 (S2 + S1), running code bfb65d2 in separate directories under terraweave_specialists_s2. Both passed CPU forward and backward checks on one real product, one station and one manual Cover window. The superseded HLS jobs 4520289 and 4520290 were cancelled. SoftCon completed as job 4520288 with epoch 5 selected.
