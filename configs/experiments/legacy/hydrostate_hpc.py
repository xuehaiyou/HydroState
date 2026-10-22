"""HPC fine-tuning against one immutable uploaded sample manifest."""

_base_ = ["./hydrostate_v1.py"]

data_root = "{{$HYDROSTATE_DATA_ROOT:/fossfs/xiaozhen/HydroState/samples_v1}}"
manifest = "{{$HYDROSTATE_MANIFEST:__data_root__}}"
if manifest == "__data_root__":
    manifest = f"{data_root}/samples.parquet"
work_dir = "{{$HYDROSTATE_WORK_DIR:/fossfs/xiaozhen/HydroState/work_dirs/hydrostate_hpc}}"

# A batch can omit a satellite modality and its encoder parameters.
find_unused_parameters = True

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    dataset=dict(manifest=manifest, data_root=data_root),
)
val_dataloader = dict(
    batch_size=2,
    num_workers=4,
    dataset=dict(manifest=manifest, data_root=data_root),
)
test_dataloader = dict(
    batch_size=2,
    num_workers=4,
    dataset=dict(manifest=manifest, data_root=data_root),
)
