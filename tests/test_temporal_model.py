"""Exercise acquisition alignment and task queries through actual OlmoEarth blocks."""

import copy
from types import SimpleNamespace

import pytest
import torch
from mmengine.config import Config

from hydrostate.cli import configure_data_contract
from hydrostate.models.data_preprocessor import HydroDataPreprocessor
from hydrostate.models.query_time import TargetTimeQuery
from hydrostate.models.temporal_model import TemporalHydroStateModel, TemporalOlmoEarthEncoder
from hydrostate.normalization import CONFIGURED_SUPPORT_VERSION
from hydrostate.tasks import task_config


def tiny_encoder(monkeypatch, tmp_path, checkpoint=False):
    from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.nn.flexi_vit import EncoderConfig
    from rslearn.models.olmoearth_pretrain import model as wrapper
    core = EncoderConfig(
        supported_modality_names=["sentinel1", "sentinel2_l2a", "landsat"],
        embedding_size=32, num_heads=4, depth=2, drop_path=0,
        num_register_tokens=0, use_flash_attn=False,
        use_linear_patch_embed=True, position_encoding="rope_3d_mixed",
    ).build()
    monkeypatch.setattr(wrapper, "load_model_from_path", lambda *a, **kw:
                        SimpleNamespace(encoder=core))
    return TemporalOlmoEarthEncoder(model_path=str(tmp_path), embedding_channels=32,
                                   gradient_checkpointing=checkpoint)


def raw_inputs():
    valid = torch.ones(2, 2, 3, dtype=torch.bool)
    valid[0, 1] = False
    # Same-day acquisitions and nearby days must remain separate.
    times = torch.tensor([[1600000000, 1600003600, 1600007200],
                          [1600086400, 1600086400, 1600086400]])[None].repeat(2, 1, 1)
    inputs = dict(sensor_valid=valid, observation_time=times)
    for key, channels in (("s1", 2), ("s2", 12), ("landsat", 11)):
        inputs[key] = torch.randn(2, 2, channels, 12, 12)
        inputs[f"{key}_valid"] = torch.ones(2, 2, 1, 12, 12, dtype=torch.bool)
    return inputs


@pytest.mark.parametrize("checkpoint", [False, True])
def test_exact_time_alignment_masks_and_encoder_gradients(monkeypatch, tmp_path, checkpoint):
    encoder = tiny_encoder(monkeypatch, tmp_path, checkpoint).train()
    inputs = raw_inputs()
    inputs["landsat_valid"][1, 0, :, :4, :4] = False
    result = encoder(inputs)
    assert result["features"].shape == (2, 6, 32, 3, 3)
    assert result["time"][1].tolist() == [1600000000, 1600086400, 1600003600,
                                         1600086400, 1600007200, 1600086400]
    assert not result["valid"][0, [1, 3, 5]].any()
    assert not result["valid"][1, 4, 0, 0]
    assert result["valid"][1, 4, 1, 1]
    result["features"][:, :, 0].square().mean().backward()
    assert all(next(block.parameters()).grad is not None for block in encoder.backbone.model.blocks)
    assert all(torch.isfinite(p.grad).all() for p in encoder.parameters() if p.grad is not None)


def test_padding_pixel_masks_and_prepared_metadata_do_not_change_features(monkeypatch, tmp_path):
    encoder = tiny_encoder(monkeypatch, tmp_path).eval()
    inputs = raw_inputs()
    inputs["landsat_valid"][:, :, :, :4, :4] = False
    changed = copy.deepcopy(inputs)
    changed["landsat"][:, :, :, :4, :4] = 123456
    alone = {key: value[:1] for key, value in inputs.items()}
    with torch.no_grad():
        batch, single, masked = encoder(inputs), encoder(alone), encoder(changed)
    torch.testing.assert_close(batch["features"][:1], single["features"], atol=3e-6, rtol=1e-5)
    torch.testing.assert_close(batch["features"], masked["features"])
    prepared = HydroDataPreprocessor(tasks=dict(lst=dict(support="pixel", grid="output")))(
        dict(inputs=inputs, data_samples={}))
    with torch.no_grad():
        torch.testing.assert_close(encoder(prepared["inputs"])["features"], batch["features"])


def test_encoder_restores_original_input_slots_and_missing_modality_batches(monkeypatch, tmp_path):
    encoder = tiny_encoder(monkeypatch, tmp_path).eval()
    inputs = raw_inputs()
    inputs["sensor_valid"][0, :, 1:] = False
    alone = {key: value[:1] for key, value in inputs.items()}
    permuted = {key: value[:, [1, 0]] for key, value in inputs.items()}
    with torch.no_grad():
        batch, single, changed = encoder(inputs), encoder(alone), encoder(permuted)
    torch.testing.assert_close(batch["features"][:1, :2], single["features"],
                               atol=3e-6, rtol=1e-5)
    for key in batch:
        torch.testing.assert_close(batch[key][:, [1, 0, 3, 2, 5, 4]], changed[key],
                                   atol=3e-6, rtol=1e-5)


def test_pooling_conditions_one_observation_and_ignores_padding():
    torch.manual_seed(31)
    pooling = TargetTimeQuery(8, ["lst", "lai"], query_channels=4)
    values = torch.randn(2, 3, 8, 2, 2, requires_grad=True)
    valid = torch.ones(2, 3, 2, 2, dtype=torch.bool)
    valid[:, 1:] = False
    valid[1, :, 0, 0] = False
    sequence = dict(features=values, valid=valid, time=torch.full((2, 3), 1600000000))
    bounds = torch.full((2, 2), 1600000000, dtype=torch.long)
    output = pooling(sequence, bounds, "lst")
    later = pooling(sequence, bounds + 86400 * 10, "lst")
    assert not torch.allclose(output, later)
    assert not output[1, :, 0, 0].any()
    altered = dict(sequence, features=values.detach().clone(), time=sequence["time"].clone())
    altered["features"][:, 1:] = 99999
    altered["time"][:, 1:] = -2**63
    torch.testing.assert_close(output, pooling(altered, bounds, "lst"))
    output.square().mean().backward()
    assert not values.grad[:, 1:].any()
    assert pooling.time_query[0].weight.grad.abs().sum() > 0


def test_new_config_keeps_all_tasks_and_explicit_time_contract():
    cfg = Config.fromfile("configs/experiments/legacy/hydrostate_ecosystem_temporal.py")
    configure_data_contract(cfg)
    assert cfg.model.type == "TemporalHydroStateModel"
    assert cfg.model.encoder.type == "TemporalOlmoEarthEncoder"
    assert cfg.train_dataloader.dataset.target_time_queries
    assert cfg.val_dataloader.dataset.target_time_queries
    assert set(cfg.model.tasks) == set(cfg.model.heads)
    assert cfg.label_packs[1].max_uncertainty.lst == 2
    assert "lst" not in cfg.label_packs[0].variables
    assert cfg.model.query_time.type == "TargetTimeQuery"


def test_shared_query_module_handles_new_task_intervals_and_permutation():
    torch.manual_seed(43)
    query = TargetTimeQuery(8, ["biomass", "runoff"], query_channels=4)
    sequence = dict(features=torch.randn(2, 3, 8, 2, 2),
                    valid=torch.ones(2, 3, 2, 2, dtype=torch.bool),
                    time=torch.tensor([[1600000000, 1600086400, 1600172800]]).repeat(2, 1))
    instant = torch.full((2, 2), 1600086400, dtype=torch.long)
    interval = instant + torch.tensor([-86400, 86400])
    # Same midpoint, different support: interval duration must reach the model.
    assert not torch.allclose(query(sequence, instant, "biomass"),
                              query(sequence, interval, "biomass"))
    permuted = {key: value[:, [2, 0, 1]] for key, value in sequence.items()}
    torch.testing.assert_close(query(sequence, interval, "runoff"),
                               query(permuted, interval, "runoff"))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = query(sequence, interval, "runoff")
        assert torch.isfinite(output).all()
        output.square().mean().backward()
    assert query.relative_score[0].weight.grad.abs().sum() > 0


def test_task_time_queries_joint_fractions_loss_and_prediction(monkeypatch, tmp_path):
    encoder = tiny_encoder(monkeypatch, tmp_path)
    tasks = {name: dict(support="pixel", grid="output") for name in ("water", "land", "lst")}
    statistics = dict(version=2, split="train", ddof=0,
                      support_version=CONFIGURED_SUPPORT_VERSION, task_specs=task_config(tasks),
                      training_fingerprint="0" * 64,
                      tasks={name: dict(count=3, mean=0, std=1) for name in tasks})
    model = TemporalHydroStateModel(
        encoder=dict(type="TemporalOlmoEarthEncoder", model_path=str(tmp_path),
                     embedding_channels=32),
        decoder=dict(type="SharedUNetDecoder", in_channels=32, channels=[8, 4, 4]),
        heads={name: dict(type="RegressionHead", in_channels=4, hidden_channels=4,
                         activation="identity") for name in tasks},
        tasks=tasks, fraction_groups=[["water", "land"]], target_statistics=statistics,
        loss=dict(type="UncertaintyWeightedNLL"), query_time=dict(query_channels=8))
    model.encoder = encoder
    model.eval()
    inputs = raw_inputs()
    inputs["target_time"] = {name: torch.full((2, 2), 1600007200) for name in tasks}
    baseline = model(inputs)
    altered = copy.deepcopy(inputs)
    altered["target_time"]["lst"] += 10 * 86400
    changed = model(altered)
    assert not torch.allclose(baseline["lst"], changed["lst"])
    torch.testing.assert_close(baseline["water"], changed["water"])
    torch.testing.assert_close(baseline["water"] + baseline["land"],
                               torch.ones_like(baseline["water"]))
    targets = dict(prediction_valid=torch.ones(2, 1, 4, 4, dtype=torch.bool))
    for name in tasks:
        targets[name] = torch.rand(2, 1, 4, 4)
        targets[name + "_valid"] = torch.ones(2, 1, 4, 4, dtype=torch.bool)
    losses = model(inputs, targets, mode="loss")
    sum(value for key, value in losses.items() if key.startswith("loss_")).backward()
    assert model.query_time.relative_score[0].weight.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    prediction = model(inputs, targets, mode="predict")[0]
    torch.testing.assert_close(prediction["target_time"]["lst"], inputs["target_time"]["lst"][0])
    altered["target_time"]["water"] += 1
    with pytest.raises(RuntimeError, match="share a target time"):
        model(altered)
