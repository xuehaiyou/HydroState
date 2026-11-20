_base_ = ['./hydrostate_joint_cgc.py']
model = dict(type='CGCAblation', intervention='pointwise')
work_dir = 'work_dirs/cgc_pointwise'
