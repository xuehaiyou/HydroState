"""TA-only control, retaining the same task query/fusion/decoder structure."""

_base_ = ["./hydrostate_mtlora.py"]
model = dict(encoder=dict(private_rank=0))
work_dir = "{{$HYDROSTATE_WORK_DIR:work_dirs/mtlora_shared}}"
