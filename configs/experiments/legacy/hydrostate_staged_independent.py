"""Ablation: keep progressive branches and auxiliary heads, disable exchange."""

_base_ = ["./hydrostate_staged.py"]

work_dir = "{{$HYDROSTATE_WORK_DIR:/fossfs/xiaozhen/HydroState/work_dirs/staged_independent}}"
model = dict(interaction=dict(mode="none"))
