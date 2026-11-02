"""Random-initialized U-Net on the identical current seven-output retrieval contract."""

_base_ = ["./hydrostate_retrieval.py"]

# Reuse all physical heads, losses and support settings; replace only the network.
model = dict(_base_.model)
model.update(
    _delete_=True,
    type="UNetHydroStateModel",
    encoder=dict(type="UNetEncoder", input_profile=_base_.input_profile,
                 channels=[64, 128, 256, 512]),
    decoder=dict(type="UNetDecoder", encoder_channels=[64, 128, 256, 512], channels=64),
    data_preprocessor=dict(type="HydroDataPreprocessor", input_profile=_base_.input_profile,
                           encode_target_season=True),
    input_fusion=None,
)
model.pop("query_time")

# Initial recipe, to be selected on validation with a comparable tuning budget.
# Keep effective batch=16, 50 epochs, shared checkpoint selection and task losses.
optim_wrapper = dict(optimizer=dict(lr=3e-4))
work_dir = "{{$HYDROSTATE_WORK_DIR:work_dirs/retrieval_unet}}"
