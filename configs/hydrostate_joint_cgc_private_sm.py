_base_ = ['./hydrostate_joint_cgc.py']
model = dict(router=dict(private_only=['sm']))
work_dir = 'work_dirs/cgc_private_sm'
