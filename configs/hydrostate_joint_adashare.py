_base_ = ['./hydrostate_joint.py']
model = dict(type='AdaShareRetrieval', branches=None, router=None,
             encoder=dict(_delete_=True, type='UTAEInput', input_profile=_base_.input_profile),
             decoder=dict(_delete_=True, type='AdaShareDecoder', groups=['coverage','sm','et']))
optim_wrapper = dict(paramwise_cfg=dict(custom_keys={'decoder.logits': dict(lr_mult=10.,decay_mult=0.)}))
work_dir = 'work_dirs/adashare_joint'
