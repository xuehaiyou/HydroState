_base_ = ['./hydrostate_joint_cgc.py']
model = dict(decoder=dict(type='LTAEStateFusion'))
work_dir = 'work_dirs/cgc_ltae'
