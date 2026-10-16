"""Existing merged_v1 samples with GEE ecosystem labels and native-pixel supervision.

Export with scripts/run_gee_labels.sh, then compute training-only target statistics.
Environment overrides also allow the same configuration on HPC.
"""

_base_ = ["./hydrostate_v1.py"]

data_root = "{{$HYDROSTATE_DATA_ROOT:/mnt/d/Hydrostate/samples/merged_v1}}"
manifest = f"{data_root}/samples.parquet"
label_pack = "{{$HYDROSTATE_LABEL_PACK:labels/ecosystem_v1}}"
label_packs = [dict(path=label_pack, max_uncertainty=dict(lst=2.0))]
statistics = "{{$HYDROSTATE_TARGET_STATS:__ecosystem__}}"
if statistics == "__ecosystem__":
    statistics = f"{data_root}/target_stats_ecosystem_v1.json"

cover_names = ["water", "trees", "grass", "flooded_vegetation", "crops",
               "shrub_and_scrub", "built", "bare", "snow_and_ice"]
tasks = dict(
    water=dict(support="pixel", grid="input", station_index=0),
    soil_moisture=dict(support="window_mean", grid="scalar", station_index=1),
    et=dict(support="pml_native", grid="native", station_index=2),
    **{name: dict(support="pixel", grid="input") for name in cover_names[1:]},
    **{name: dict(support="native", grid="native")
       for name in ("lai", "gpp", "albedo_bsa", "albedo_wsa")},
    lst=dict(support="pixel", grid="output"),
)

model = dict(
    heads=dict(
        **{name: dict(type="RegressionHead", in_channels=128, activation="identity")
           for name in cover_names},
        **{name: dict(type="RegressionHead", in_channels=128, activation="softplus")
           for name in ("lai", "gpp")},
        **{name: dict(type="RegressionHead", in_channels=128, activation="sigmoid")
           for name in ("albedo_bsa", "albedo_wsa")},
        lst=dict(type="RegressionHead", in_channels=128, activation="identity"),
    ),
    fraction_groups=[cover_names],
    target_statistics=statistics,
)

train_dataloader = dict(dataset=dict(manifest=manifest, data_root=data_root))
val_dataloader = dict(dataset=dict(manifest=manifest, data_root=data_root))
test_dataloader = dict(dataset=dict(manifest=manifest, data_root=data_root))
work_dir = "work_dirs/hydrostate_ecosystem"
