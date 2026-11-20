_base_ = ['./hydrostate_joint_cgc.py']
model = dict(router=dict(spatial_gates=['sm']))
work_dir = 'work_dirs/cgc_local_sm'
