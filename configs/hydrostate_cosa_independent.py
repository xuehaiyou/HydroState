"""Experiment B: identical CoSA modules, with cross-expert transfer disabled."""

_base_ = ["./hydrostate_cosa.py"]
model = dict(router=dict(enabled=False))
work_dir = "{{$HYDROSTATE_WORK_DIR:work_dirs/cosa_independent}}"
