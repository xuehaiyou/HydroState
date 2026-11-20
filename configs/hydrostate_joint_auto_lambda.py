_base_ = ['./hydrostate_joint_cgc.py']
model = dict(type='AutoLambdaCGC', meta_plan='{{$HYDROSTATE_META_PLAN:__required_meta_plan__}}')
