"""MLoRE block candidate on the same interval-conditioned CoSA feature path."""

_base_ = ["./hydrostate_joint.py"]

model = dict(router=dict(
    _delete_=True, type="MLoRERouter", grid_size=30,
    ranks=(4, 8, 12, 16), private_rank=8, topk=2, pre_softmax=True))
work_dir = "{{$HYDROSTATE_WORK_DIR:work_dirs/mlore_joint}}"
