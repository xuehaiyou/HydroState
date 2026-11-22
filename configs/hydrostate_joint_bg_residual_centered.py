_base_ = ['./hydrostate_joint_bg_residual.py']
model = dict(router=dict(center=True))
work_dir = 'work_dirs/bg_residual_centered'
