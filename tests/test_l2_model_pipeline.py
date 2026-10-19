"""L2 samples, gradients, temporal metadata and regional checkpoint contracts."""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from mmengine.config import Config

from hydrostate.cli import configure_data_contract
from hydrostate.data.input_products import L2_PROFILE, landsat_l2_spec
from hydrostate.data.model_inputs import prepare_model_inputs
from hydrostate.datasets.zarr_dataset import HydroStateZarrDataset, hydro_collate
from hydrostate.inference.predict import RegionDataset, load_model, masked_outputs
from hydrostate.models.data_preprocessor import HydroDataPreprocessor
from hydrostate.registry import MODELS
from hydrostate.storage import SampleShardWriter, Schema, VariablePackWriter


@pytest.fixture
def l2_samples(tmp_path):
    schema = Schema(time_steps=2, height=12, width=12)
    writer = SampleShardWriter(str(tmp_path / 'inputs.zarr'), str(tmp_path / 'labels.zarr'),
                               2, schema=schema)
    for i in range(2):
        inputs = {k: np.zeros(v.tail_shape, dtype=v.dtype) for k, v in schema.inputs.items()}
        inputs['sensor_valid'][:] = True  # Deliberately stale L1 availability.
        inputs['observation_time'][:] = 1600000000
        inputs['observation_time'][1] += 86400
        inputs['s1_valid'][:] = i == 1
        inputs['landsat_valid'][:] = True
        labels = {k: np.zeros(v.tail_shape, dtype=v.dtype) for k, v in schema.targets.items()}
        labels['water'][:] = labels['soil_moisture'][:] = 0.2 + i * 0.1
        labels['water_valid'][:] = labels['soil_moisture_valid'][:] = True
        writer.write(inputs, labels)
    writer.finalize()
    table = pd.DataFrame([dict(sample_id=str(i), split='train', input_shard='inputs.zarr',
                               input_row=i, label_shard='labels.zarr', label_row=i,
                               target_time='2020-09-14') for i in range(2)])
    manifest = tmp_path / 'samples.parquet'
    table.to_parquet(manifest, index=False)
    spec = dict(landsat_l2_spec(), shape=[2, 7, 12, 12])
    with VariablePackWriter(tmp_path / 'l2', manifest, kind='input',
                            variables=dict(landsat_l2=spec), source='fixture', version='1') as pack:
        for i in range(2):
            valid = np.full((2, 7, 12, 12), i == 0)
            valid[0, 3, :4, :4] = False
            pack.write(str(i), dict(landsat_l2=np.full(valid.shape, -0.05),
                                    landsat_l2_valid=valid,
                                    landsat_l2_time=np.array([1600007200, 1600093600])))
        pack.finalize()
    return HydroStateZarrDataset(str(manifest), 'train', input_profile=L2_PROFILE,
                                 input_packs=[dict(path='l2')])


def test_l2_replaces_l1_masks_times_and_spatial_support(l2_samples):
    a, b = l2_samples[0], l2_samples[1]
    assert 'landsat' not in a['inputs']
    assert a['inputs']['observation_time'][:, 2].tolist() == [1600007200, 1600093600]
    assert not b['inputs']['sensor_valid'][:, 2].any()
    assert not b['inputs']['observation_time'][:, 2].any()
    assert a['inputs']['landsat_l2'][0, :, 5, 5].max() < 0
    assert not a['inputs']['landsat_l2_valid'][0, :, :4, :4].any()
    batch = hydro_collate([a, b])
    batch['inputs']['landsat_l2_valid'][0, 1] = False
    data = HydroDataPreprocessor(input_profile=L2_PROFILE)(batch)
    assert data['data_samples']['prediction_valid'].shape == (2, 1, 4, 4)
    assert not data['data_samples']['prediction_valid'][0, 0, 0, 0]
    assert data['data_samples']['prediction_valid'][1].all()
    assert not data['inputs']['sensor_valid'][0, 1, 2]
    with pytest.raises(ValueError, match='explicit L2'):
        HydroDataPreprocessor()(hydro_collate([a]))
    with pytest.raises(ValueError, match='not accompany'):
        prepare_model_inputs(dict(a['inputs'], landsat=torch.zeros(1)), L2_PROFILE)
    bad = copy.deepcopy(a['inputs'])
    bad['landsat_l2_time'][:] = 0
    with pytest.raises(ValueError, match='positive acquisition'):
        prepare_model_inputs(bad, L2_PROFILE)


def test_uncollected_l2_records_fail_before_training(l2_samples):
    table = pd.read_parquet(l2_samples.manifest_path)
    extra = dict(table.iloc[0], sample_id='not-collected')
    pd.concat([table, pd.DataFrame([extra])]).to_parquet(l2_samples.manifest_path, index=False)
    with pytest.raises(ValueError, match='no records for 1 train samples'):
        HydroStateZarrDataset(str(l2_samples.manifest_path), 'train', input_profile=L2_PROFILE,
                              input_packs=[dict(path='l2')])


def test_mask_only_statistics_match_full_image_reads(l2_samples, monkeypatch):
    from hydrostate.tasks import task_config
    from tools.data import compute_target_statistics as statistics

    tasks = {k: v for k, v in task_config().items() if k != 'et'}
    options = dict(tasks=tasks, input_profile=L2_PROFILE, input_packs=[dict(path='l2')],
                   num_workers=0)
    fast = statistics.compute_statistics(l2_samples.manifest_path, l2_samples.data_root, **options)
    monkeypatch.setattr(statistics, '_SupportDataset', HydroStateZarrDataset)
    full = statistics.compute_statistics(l2_samples.manifest_path, l2_samples.data_root, **options)
    assert fast['tasks'] == full['tasks']
    assert fast['training_fingerprint'] == full['training_fingerprint']


def tiny_backbone(monkeypatch):
    from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.nn.flexi_vit import EncoderConfig
    from rslearn.models.olmoearth_pretrain import model as wrapper

    def load(*args, **kwargs):
        return SimpleNamespace(encoder=EncoderConfig(
            supported_modality_names=['sentinel1', 'sentinel2_l2a', 'landsat'],
            embedding_size=32, num_heads=4, depth=3, drop_path=0, num_register_tokens=0,
            use_flash_attn=False, use_linear_patch_embed=True,
            position_encoding='rope_3d_mixed').build())

    monkeypatch.setattr(wrapper, 'load_model_from_path', load)


def test_l2_gradient_checkpoint_reload_and_regional_queries(
        l2_samples, monkeypatch, tmp_path, target_statistics):
    from mmengine.registry import init_default_scope

    init_default_scope('hydrostate')
    tiny_backbone(monkeypatch)
    cfg = Config.fromfile('configs/hydrostate_multilevel_temporal_l2.py')
    configure_data_contract(cfg)
    cfg.model.target_statistics = target_statistics
    cfg.model.encoder.update(model_path=str(tmp_path), embedding_channels=32,
                             out_indices=[0, 1, 2], gradient_checkpointing=True)
    cfg.model.decoder.update(in_channels=32, channels=[8, 8, 4])
    cfg.model.interaction.channels = 4
    for head in cfg.model.heads.values():
        head.update(in_channels=4, hidden_channels=4)
    model = MODELS.build(cfg.model)
    batch = hydro_collate([l2_samples[0], l2_samples[1]])
    batch['inputs']['target_time'] = {
        k: torch.tensor([[1600086400, 1600086400]] * 2) for k in model.tasks}
    data = model.data_preprocessor(batch, training=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    before = model.encoder.landsat_l2_adapter.projection.weight.detach().clone()
    loss = model(**data, mode='loss')
    sum(v for k, v in loss.items() if k.startswith('loss_')).backward()
    gradient = model.encoder.landsat_l2_adapter.projection.weight.grad
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    assert model.query_time.relative_score[0].weight.grad.abs().sum() > 0
    optimizer.step()
    assert not torch.equal(before, model.encoder.landsat_l2_adapter.projection.weight)
    model.eval()
    with torch.no_grad():
        expected = model(data['inputs'])
    checkpoint = tmp_path / 'model.pth'
    torch.save(dict(state_dict=model.state_dict()), checkpoint)
    statistics = tmp_path / 'stats.json'
    target_statistics['data_contract'] = dict(input_profile=L2_PROFILE)
    statistics.write_text(json.dumps(target_statistics))
    config_file = tmp_path / 'model.py'
    Config(dict(model=cfg.model, input_profile=L2_PROFILE)).dump(str(config_file))
    plan = dict(config=dict(input_profile=L2_PROFILE, pretrained=str(tmp_path),
                            target_statistics=str(statistics)))
    restored, _ = load_model(plan, checkpoint=checkpoint, model_config=config_file, device='cpu')
    with torch.no_grad():
        actual = restored(data['inputs'])
    for task in expected:
        torch.testing.assert_close(expected[task], actual[task])
    # The regional reader supplies explicit instant WF/SM and interval ET queries.
    arrays = {k: v.numpy() for k, v in l2_samples[0]['inputs'].items()}
    monkeypatch.setattr('hydrostate.inference.predict.assemble_inputs', lambda *a, **k: arrays)
    region = RegionDataset(tmp_path, plan, pd.DataFrame([dict(row=0, col=0)]),
                           query_period=dict(time='2020-09-14', start='2020-09-10',
                                             end='2020-09-18'))
    region.cache = object()
    inputs = hydro_collate([dict(inputs=region[0]['inputs'], data_samples={})])['inputs']
    assert inputs['target_time']['et'][0, 1] - inputs['target_time']['et'][0, 0] == 8 * 86400
    with torch.no_grad():
        result = masked_outputs(restored, inputs)
    assert all(v.shape == (1, 4, 4) and np.isfinite(v).all() for v in result.values())
    state = model.state_dict()
    state = {k: v for k, v in state.items() if not k.startswith('encoder.landsat_l2_adapter.')}
    torch.save(dict(state_dict=state), checkpoint)
    with pytest.raises(RuntimeError, match='landsat_l2_adapter'):
        load_model(plan, checkpoint=checkpoint, model_config=config_file, device='cpu')


@pytest.mark.parametrize('name,count', [('multilevel_temporal', 3),
                                       ('ecosystem_multilevel_temporal', 16)])
def test_l2_configs_preserve_tasks_and_architecture(name, count):
    cfg = Config.fromfile(f'configs/hydrostate_{name}_l2.py')
    configure_data_contract(cfg)
    assert len(cfg.model.heads) == count
    assert cfg.model.encoder.out_indices == [3, 7, 11]
    assert cfg.model.decoder.type == 'MultiLevelFusionDecoder'
    assert cfg.model.adapter.enabled and cfg.target_time_queries
    assert cfg.model.encoder.input_profile == L2_PROFILE
    assert cfg.model.data_preprocessor.input_profile == L2_PROFILE
    assert cfg.train_dataloader.dataset.input_profile == L2_PROFILE
    assert cfg.train_dataloader.batch_size * cfg.optim_wrapper.accumulative_counts == 16
