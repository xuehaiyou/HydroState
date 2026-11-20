_base_ = ['./hydrostate_joint_cgc.py']
model = dict(type='CGCAblation', intervention='constant_gate')
work_dir = 'work_dirs/cgc_constant_gate'
