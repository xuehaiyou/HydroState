"""Two spatial CGC blocks with one learned local/global mixture per task group."""

_base_ = ['./hydrostate_joint_cgc.py']
model = dict(router=dict(type='RecurrentCGCRouter', spatial_gates=[], learn_locality=True, depth=2))
work_dir = 'work_dirs/adaptive_cgc'
