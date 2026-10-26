"""Current seven-output full fine-tuning with the historical multilevel LLRD recipe."""

_base_ = ["./hydrostate_retrieval.py"]

model = dict(encoder=dict(finetuning="full", shared_rank=0, private_rank=0))

# Match hydrostate_ecosystem_multilevel_temporal_hpc's full-fine-tuning recipe.
# The current data contract, task decoders, effective batch and schedule are inherited.
optim_wrapper = dict(
    optimizer=dict(weight_decay=0.05),
    constructor="OlmoEarthLayerDecayOptimWrapperConstructor",
    paramwise_cfg=dict(num_layers=12, layer_decay_rate=0.65, recipe="official"),
)
work_dir = "{{$HYDROSTATE_WORK_DIR:work_dirs/retrieval_full}}"
