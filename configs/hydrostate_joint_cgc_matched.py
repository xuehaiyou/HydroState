"""Matched single extraction level; same expert/gate family as two-level PLE."""

_base_ = ['./hydrostate_joint_ple.py']
model = dict(decoder=dict(widths=(256,)))
work_dir = '{{$HYDROSTATE_WORK_DIR:work_dirs/cgc_matched_joint}}'
