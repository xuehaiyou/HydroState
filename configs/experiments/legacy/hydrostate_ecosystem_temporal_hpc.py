"""All 16 retrieval tasks with acquisition-time queries on the merged HPC dataset."""

_base_ = ["hydrostate_ecosystem_temporal.py"]

data_root = "{{$HYDROSTATE_DATA_ROOT:/scr/u/xiaoz/HydroState/samples_merged_v1}}"
manifest = "{{$HYDROSTATE_MANIFEST:__merged__}}"
if manifest == "__merged__":
    manifest = f"{data_root}/manifests/merged_v1_n20000/samples.parquet"
statistics = "{{$HYDROSTATE_TARGET_STATS:__temporal_hpc__}}"
if statistics == "__temporal_hpc__":
    statistics = manifest.rsplit("/", 1)[0] + "/target_stats_ecosystem_temporal_v1.json"
work_dir = "{{$HYDROSTATE_WORK_DIR:/scr/u/xiaoz/HydroState/work_dirs/ecosystem_temporal}}"

model = dict(
    target_statistics=statistics,
    encoder=dict(
        model_path="{{$HYDROSTATE_MODEL_PATH:/scr/u/xiaoz/HydroState/pretrained/OlmoEarth-v1_2-Base}}",
    ),
)

# Exact acquisition slots and eight query groups need more memory than the
# original three-task model. Keep effective batch 16 on a single 48 GB GPU.
train_dataloader = dict(
    batch_size=2, num_workers=4,
    dataset=dict(manifest=manifest, data_root=data_root),
)
val_dataloader = dict(
    batch_size=2, num_workers=4,
    dataset=dict(manifest=manifest, data_root=data_root),
)
test_dataloader = dict(
    batch_size=2, num_workers=4,
    dataset=dict(manifest=manifest, data_root=data_root),
)
optim_wrapper = dict(accumulative_counts=8)
# Inherited 1000-microstep warmup gives 125 optimizer updates.
custom_hooks = [
    dict(type="HydroVisualizationHook", num_samples=2, interval=1,
         out_dir="visualizations", et_vmax=10.0),
]
