"""MTLoRA control with frozen input patch embedding."""

_base_ = ["./hydrostate_mtlora.py"]
model = dict(encoder=dict(train_patch_embedding=False))
work_dir = "{{$HYDROSTATE_WORK_DIR:work_dirs/mtlora_frozen_embed}}"
