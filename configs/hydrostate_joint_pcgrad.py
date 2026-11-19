_base_ = ['./hydrostate_joint_cgc.py']
model = dict(type='GradientCGC', gradient_method='pcgrad')
work_dir = 'work_dirs/PCGrad'
