"""Causal controls: gradients, fixed exposure, temporal support and token layouts."""
import copy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from hydrostate.experiments.sm_causal import (
    CausalHydroStateModel, CausalNLL, FixedGlobalBatchSampler,
    filter_temporal_inputs, temporal_signature,
)
from hydrostate.normalization import TASKS
from hydrostate.registry import MODELS


def test_mse_value_gradient_and_empty_task(target_statistics):
    loss = CausalNLL(objective='mse', target_statistics=target_statistics)
    p = torch.tensor([.1,.3,float('nan')], requires_grad=True)
    y = torch.tensor([.2,.25,float('nan')])
    got, count = loss('soil_moisture',p,y,torch.tensor([True,True,False]))
    expected = ((p[:2]-y[:2])/.1).square().mean()
    torch.testing.assert_close(got,expected)
    torch.testing.assert_close(torch.autograd.grad(got,p,retain_graph=True)[0],torch.autograd.grad(expected,p)[0])
    empty, count = loss('soil_moisture',p,y,torch.zeros(3,dtype=torch.bool))
    assert empty == 0 and count == 0
    assert all(not v.requires_grad for v in loss.log_variances.values())


def test_single_task_keeps_predictions_but_removes_auxiliary_gradients(target_statistics):
    class Encoder(torch.nn.Module):
        def forward(self,x):
            return x
    MODELS.register_module(name='CausalTestEncoder',module=Encoder,force=True)
    kwargs = dict(encoder=dict(type='CausalTestEncoder'),
        decoder=dict(type='SharedUNetDecoder',in_channels=4,channels=[4],scale_factor=1),
        heads={t:dict(type='RegressionHead',in_channels=4,activation='sigmoid') for t in TASKS},
        loss=dict(type='CausalNLL',objective='mse'),target_statistics=target_statistics,output_scale=1)
    torch.manual_seed(1)
    joint = CausalHydroStateModel(**kwargs)
    torch.manual_seed(1)
    single = CausalHydroStateModel(active_tasks=['soil_moisture'],**kwargs)
    x = torch.randn(2,4,3,3)
    for key in TASKS:
        torch.testing.assert_close(joint._forward(x)[key],single._forward(x)[key])
    samples = dict(water=torch.zeros(2,1,3,3),water_valid=torch.ones(2,1,3,3,dtype=torch.bool),
        soil_moisture=torch.ones(2,1,1,1)*.2,soil_moisture_valid=torch.ones(2,1,1,1,dtype=torch.bool),
        et_native_weights=torch.ones(2,1,3,3)/9,et_native_value=torch.ones(2,1),
        et_native_valid=torch.ones(2,1,dtype=torch.bool))
    losses = single.loss(x,samples)
    assert set(k for k in losses if k.startswith('loss_')) == {'loss_soil_moisture'}
    losses['loss_soil_moisture'].backward()
    assert all(p.grad is None and not p.requires_grad for p in single.heads['water'].parameters())
    assert any(p.grad is not None for p in single.heads['soil_moisture'].parameters())
    assert any(p.grad is not None for p in single.decoder.parameters())


def test_fixed_schedule_reconstructs_global_batches(tmp_path,monkeypatch):
    import hydrostate.experiments.sm_causal as module
    expected = np.arange(64,dtype=np.int32).reshape(2,32)
    path = tmp_path/'schedule.npy'
    np.save(path,expected)
    streams = []
    for rank in (0,1):
        monkeypatch.setattr(module,'get_dist_info',lambda rank=rank:(rank,2))
        sampler = FixedGlobalBatchSampler(list(range(64)),path)
        streams.append(np.array(list(sampler)).reshape(2,16))
    np.testing.assert_array_equal(np.concatenate(streams,axis=1),expected)


def test_temporal_filter_removes_future_and_shares_recent_support():
    day = 86400*100
    inputs = dict(sensor_valid=torch.ones(4,3,dtype=torch.bool),
                  observation_time=torch.tensor([[day-86400*20]*3,[day-86400*10]*3,[day]*3,[day+86400]*3]))
    for key in ('s1','s2','landsat'):
        inputs[key] = torch.ones(4,2,6,6)
        inputs[key+'_valid'] = torch.ones(4,1,6,6,dtype=torch.bool)
        inputs[key+'_valid'][2,:,:,0] = False
    history,support_h = filter_temporal_inputs(copy.deepcopy(inputs),day,'history')
    recent,support_r = filter_temporal_inputs(copy.deepcopy(inputs),day,'recent')
    assert history['sensor_valid'].sum()==9 and recent['sensor_valid'].sum()==3
    torch.testing.assert_close(support_h,support_r)
    assert not support_h[:,:,0].any()
    assert history['s1'][3].sum()==0 and not history['s1_valid'][3].any()
    assert recent['s1'][:2].sum()==0


def test_signature_matches_upstream_preparation():
    from datetime import datetime,timezone,timedelta
    from rslearn.models.olmoearth_pretrain.model import OlmoEarth,MODALITY_NAMES
    from rslearn.train.model_context import ModelContext,RasterImage
    from hydrostate.models.olmoearth_encoder import MODALITIES
    order = [MODALITIES.index(m) for m in MODALITY_NAMES if m in MODALITIES]
    wrapper = SimpleNamespace(timestamp_error_tolerance=timedelta(days=15),
                              datetimes_to_timestamps=OlmoEarth.datetimes_to_timestamps)
    rng = np.random.default_rng(8)
    for _ in range(12):
        times = (np.arange(4)[:,None]*86400*16 + rng.integers(0,86400*5,size=(4,3)) + 1600000000).tolist()
        valid = (rng.random((4,3))>.35).tolist()
        if not np.any(valid):
            continue
        context = {}
        for mi,m in enumerate(MODALITIES):
            selected = [t for t in range(4) if valid[t][mi]]
            if not selected:
                continue
            dates = [datetime.fromtimestamp(times[t][mi],tz=timezone.utc).replace(tzinfo=None) for t in selected]
            context[m] = RasterImage(torch.zeros([2,12,11][mi],len(selected),4,4),timestamps=[(d,d) for d in dates])
        sample,modalities,_ = OlmoEarth._prepare_modality_inputs(wrapper,ModelContext(inputs=[context],metadatas=[]))
        from rslearn.models.olmoearth_pretrain.model import MaskValue
        expected=[]
        for m in modalities:
            mask=getattr(sample,m+'_mask')[0,0,0,:,0]
            expected.append((MODALITIES.index(m),tuple(torch.where(mask != MaskValue.MISSING.value)[0].tolist())))
        assert temporal_signature(times,valid,15*86400,order)==tuple(expected)


def test_preparation_schedules_match_and_have_global_sm():
    from tools.experiments.prepare_sm_causal import schedule
    pool=pd.DataFrame({'sample_id':[f's{i}' for i in range(100)]})
    left,_=schedule(pool,{'s0'},100,3407)
    right,_=schedule(pool,{'s0'},100,3407)
    np.testing.assert_array_equal(left,right)
    assert (left==0).any(axis=1).all()
    with pytest.raises(ValueError):
        schedule(pool,set(),2,3407)


def test_paired_bootstrap_sign():
    from tools.experiments.summarize_sm_causal import paired
    a=pd.DataFrame(dict(task=['soil_moisture']*4,sample_id=list('abcd'),count=[1]*4,
                        block=['a','a','b','b'],sq=[4.]*4,y=[.2]*4))
    b=a.copy()
    b['sq']=1.
    result=paired(a,b)
    assert result['candidate_minus_reference']==-1
    assert result['delta_ci95']==[-1.,-1.]
    b.loc[0,'y']=.3
    with pytest.raises(ValueError):
        paired(a,b)
