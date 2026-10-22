"""Ablation: normalize the two incoming task gates with softmax."""

_base_ = ["hydrostate_staged.py"]

work_dir = "{{$HYDROSTATE_WORK_DIR:/fossfs/xiaozhen/HydroState/work_dirs/staged_softmax}}"
model = dict(interaction=dict(mode="softmax"))
