"""One spatial CGC block with the same learned gate locality as the two-block model."""

_base_ = ['./hydrostate_joint_adaptive_cgc.py']
model = dict(router=dict(depth=1))
work_dir = 'work_dirs/adaptive_cgc_single'
