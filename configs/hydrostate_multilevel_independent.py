"""Matched multilevel ablation: task adapters without cross-task messages."""

_base_ = ["./hydrostate_multilevel.py"]

work_dir = "{{$HYDROSTATE_WORK_DIR:/fossfs/xiaozhen/HydroState/work_dirs/multilevel_independent}}"
model = dict(interaction=dict(mode="none"))
