"""Matched joint-supervision CoSA ablation: disable only expert interaction."""

_base_ = ["./hydrostate_joint.py"]

model = dict(router=dict(enabled=False))
work_dir = "{{$HYDROSTATE_WORK_DIR:work_dirs/cosa_joint_independent}}"
