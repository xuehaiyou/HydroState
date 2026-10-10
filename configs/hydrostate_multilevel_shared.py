"""Matched multilevel ablation: shared features directly feed the three heads."""

_base_ = ["./hydrostate_multilevel.py"]

work_dir = "{{$HYDROSTATE_WORK_DIR:/fossfs/xiaozhen/HydroState/work_dirs/multilevel_shared}}"
model = dict(adapter=dict(enabled=False), interaction=dict(mode="none"))
