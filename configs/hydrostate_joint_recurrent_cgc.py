"""Spatial CGC extraction followed by two weight-tied residual refinements."""

_base_ = ['./hydrostate_joint_cgc_local_sm.py']
model = dict(router=dict(type='RecurrentCGCRouter', depth=3))
work_dir = 'work_dirs/recurrent_cgc'
