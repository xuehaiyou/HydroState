_base_ = ['./hydrostate_joint_cgc.py']
model = dict(type='GradientCGC', gradient_method='aligned')
work_dir = 'work_dirs/Aligned'
