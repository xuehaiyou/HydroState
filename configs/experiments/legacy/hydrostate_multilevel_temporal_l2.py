"""WF/SM/ET: Landsat L2 SR, multilevel decoding and actual-time queries."""

_base_ = ["hydrostate_multilevel.py"]

input_profile = "landsat_l2_sr_v1"
input_packs = [dict(path="{{$HYDROSTATE_L2_PACK:inputs/landsat_l2_sr_v1}}",
                    variables=["landsat_l2"])]
target_time_queries = True
data_root = "{{$HYDROSTATE_DATA_ROOT:/mnt/d/Hydrostate/samples/merged_v1}}"
manifest = "{{$HYDROSTATE_MANIFEST:__root__}}"
if manifest == "__root__":
    manifest = f"{data_root}/samples.parquet"
statistics = "{{$HYDROSTATE_TARGET_STATS:__l2__}}"
if statistics == "__l2__":
    statistics = manifest.rsplit("/", 1)[0] + "/target_stats_multilevel_temporal_l2_v1.json"
work_dir = "{{$HYDROSTATE_WORK_DIR:work_dirs/multilevel_temporal_l2}}"

model = dict(
    type="MultiLevelTemporalHydroStateModel",
    encoder=dict(type="MultiLevelTemporalOlmoEarthEncoder", input_profile=input_profile),
    data_preprocessor=dict(input_profile=input_profile),
    query_time=dict(type="TargetTimeQuery", query_channels=64),
    target_statistics=statistics,
    # Product WF and SM observations can have different query dates in training.
    water_soil_coupling=None,
)
train_dataloader = dict(batch_size=2, dataset=dict(manifest=manifest, data_root=data_root))
val_dataloader = dict(batch_size=2, dataset=dict(manifest=manifest, data_root=data_root))
test_dataloader = dict(batch_size=2, dataset=dict(manifest=manifest, data_root=data_root))
optim_wrapper = dict(accumulative_counts=8)
param_scheduler = [
    dict(type="LinearLR", start_factor=0.01, by_epoch=False, begin=0, end=1000),
    dict(type="CosineAnnealingLR", by_epoch=True, begin=0, end=50, T_max=50,
         eta_min_ratio=0.01),
]
