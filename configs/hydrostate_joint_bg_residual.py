_base_ = ['./hydrostate_joint.py']
model = dict(type='BackgroundResidualModel', router=dict(_delete_=True, center=False))
work_dir = 'work_dirs/bg_residual'
