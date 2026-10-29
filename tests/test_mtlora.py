"""Real tiny OlmoEarth checks for routing, masks, config and training contracts."""

import copy
from types import SimpleNamespace

import pytest
import torch
from mmengine.config import Config
from mmengine.optim import build_optim_wrapper
from mmengine.registry import init_default_scope
from test_multilevel_temporal_model import moments, raw_inputs
from test_shared_private_model import samples

from hydrostate.cli import configure_data_contract
from hydrostate.models.mtlora import AlignedLandCoupling, MTLoRAExitBlock, MTLoRALinear
from hydrostate.models.multilevel_temporal_model import MultiLevelTemporalOlmoEarthEncoder
from hydrostate.models.shared_private_lora import LowRankUpdate
from hydrostate.registry import MODELS
from hydrostate.tasks import LEGACY_TASKS, task_config


@pytest.fixture(autouse=True)
def tiny(monkeypatch):
    from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.nn.flexi_vit import EncoderConfig
    from rslearn.models.olmoearth_pretrain import model as wrapper

    def load(*args, **kwargs):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(51)
            core = EncoderConfig(
                supported_modality_names=["sentinel1", "sentinel2_l2a", "landsat"],
                embedding_size=32,
                num_heads=4,
                depth=6,
                drop_path=0,
                num_register_tokens=2,
                use_flash_attn=False,
                use_linear_patch_embed=True,
                position_encoding="rope_3d_mixed",
            ).build()
        return SimpleNamespace(encoder=core)

    monkeypatch.setattr(wrapper, "load_model_from_path", load)
    init_default_scope("hydrostate")
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def config(tmp_path, checkpoint=False):
    cfg = Config.fromfile("configs/hydrostate_mtlora.py")
    configure_data_contract(cfg)
    cfg.model.encoder.update(
        model_path=str(tmp_path),
        embedding_channels=32,
        out_indices=[1, 3, 5],
        gradient_checkpointing=checkpoint,
        shared_rank=2,
        private_rank=2,
    )
    cfg.model.decoder.update(in_channels=32, channels=8)
    for head in cfg.model.heads.values():
        head.in_channels = 8
    # Generic routing tests use three tasks; the production seven-head contract
    # is exercised end-to-end in tests/retrieval/test_model.py.
    cfg.model.tasks = task_config(LEGACY_TASKS)
    cfg.model.heads = {
        name: dict(type="RegressionHead", in_channels=8, activation=activation)
        for name, activation in (
            ("water", "sigmoid"), ("soil_moisture", "sigmoid"), ("et", "softplus"))}
    cfg.model.encoder.input_profile = "olmoearth_l1_v1"
    cfg.model.data_preprocessor.input_profile = "olmoearth_l1_v1"
    cfg.model.input_fusion = None
    cfg.model.fraction_groups = []
    cfg.model.water_soil_coupling = dict(enabled=True, min_land_fraction=.01,
                                       min_window_land_fraction=.01)
    cfg.model.land_time_policy = "aligned_or_window"
    cfg.model.target_statistics = moments(task_config(LEGACY_TASKS))
    return cfg


@pytest.mark.parametrize("private_rank", [0, 2])
def test_shared_feature_reuse_matches_independent_computation_and_gradients(tmp_path, private_rank):
    cfg = config(tmp_path)
    cfg.model.encoder.private_rank = private_rank
    cfg.model.input_fusion = dict(type="AuxiliaryInputFusion", feature_channels=32, inputs=dict(
        quality=dict(channels=2, mean=[0, 0], std=[1, 1], temporal="mean")))
    model = MODELS.build(cfg.model).train()
    reference = copy.deepcopy(model)
    # Equal-valued, distinct branch tensors reproduce the original per-task
    # restoration, preventing identity-based reuse without changing the function.
    for block in reference.encoder.backbone.model.blocks:
        if isinstance(block, MTLoRAExitBlock):
            original = block.forward
            def separate(*args, _forward=original, **kwargs):
                shared, *branches = _forward(*args, **kwargs)
                return (shared, *(value.clone() for value in branches))
            block.forward = separate
    original_fusion = reference.input_fusion.forward
    def uncached(features, inputs, **kwargs):
        return original_fusion(features, inputs)
    reference.input_fusion.forward = uncached
    inputs = raw_inputs(model.tasks)
    inputs["quality"] = torch.randn(2, 2, 2, 12, 12)
    inputs["quality_valid"] = torch.rand_like(inputs["quality"]) > 0.3
    optimized, expected = model(inputs), reference(inputs)
    for task in model.tasks:
        torch.testing.assert_close(optimized[task], expected[task], atol=1e-6, rtol=1e-5)
    sum(v.square().mean() for v in optimized.values()).backward()
    sum(v.square().mean() for v in expected.values()).backward()
    for (name, left), (_, right) in zip(model.named_parameters(), reference.named_parameters()):
        if left.grad is None:
            assert right.grad is None, name
        else:
            torch.testing.assert_close(left.grad, right.grad, atol=2e-6, rtol=2e-5, msg=name)
    # Cached encodings must not survive into the next forward with changed input.
    inputs["quality"] += 2
    for task, value in model(inputs).items():
        torch.testing.assert_close(value, reference(inputs)[task], atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("checkpoint", [False, True])
def test_single_attention_path_and_private_gradient_isolation(tmp_path, checkpoint):
    model = MODELS.build(config(tmp_path, checkpoint).model).train()
    assert model.encoder.landsat_l2_adapter is None
    x = raw_inputs(model.tasks)
    calls = []
    handles = [
        b.attn.register_forward_hook(lambda *args: calls.append(1))
        for b in model.encoder.backbone.model.blocks
    ]
    output = model(x)
    assert len(calls) == 6  # Six blocks, not six times the number of variables.
    assert all(y.shape == (2, 1, 4, 4) for y in output.values())
    output["water"].square().mean().backward()
    for handle in handles:
        handle.remove()
    for module in model.encoder.modules():
        if isinstance(module, MTLoRALinear):
            assert module.base.weight.grad is None
            for task, branch in module.private.items():
                if task == "water":
                    assert branch.b.weight.grad is not None
                    assert branch.b.weight.grad.abs().sum() > 0
                else:
                    assert branch.b.weight.grad is None
    assert model.encoder.backbone.model.blocks[0].attn.q.shared.b.weight.grad.abs().sum() > 0
    assert next(model.decoders["soil_moisture"].parameters()).grad is None
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_zero_adapters_match_original_and_private_exits_do_not_feed_back(tmp_path):
    cfg = config(tmp_path)
    encoder = MODELS.build(dict(cfg.model.encoder, tasks=task_config(LEGACY_TASKS))).eval()
    original = MultiLevelTemporalOlmoEarthEncoder(
        model_path=str(tmp_path), embedding_channels=32, out_indices=[1, 3, 5]
    ).eval()
    x = raw_inputs()
    with torch.no_grad():
        reference, baseline = original(x), encoder(x)
    for task in encoder.tasks:
        for a, b in zip(reference, baseline[task]):
            torch.testing.assert_close(a["features"], b["features"])
    block = encoder.backbone.model.blocks[1]
    torch.nn.init.normal_(block.projection.private["water"].b.weight, std=0.1)
    with torch.no_grad():
        changed = encoder(x)
    assert not torch.allclose(changed["water"][0]["features"], baseline["water"][0]["features"])
    for task in encoder.tasks:
        for depth in range(3):
            if task == "water" and depth == 0:
                continue
            torch.testing.assert_close(
                changed[task][depth]["features"], baseline[task][depth]["features"]
            )


def test_mask_batch_invariance_query_independence_and_checkpoint(tmp_path):
    model = MODELS.build(config(tmp_path, True).model).eval()
    x = raw_inputs(model.tasks)
    x["landsat_valid"][:, :, :, :4, :4] = False
    altered = copy.deepcopy(x)
    altered["landsat"][:, :, :, :4, :4] = 1e6
    single = {
        k: ({t: z[:1] for t, z in v.items()} if isinstance(v, dict) else v[:1])
        for k, v in x.items()
    }
    with torch.no_grad():
        baseline, masked, alone = model(x), model(altered), model(single)
        altered = copy.deepcopy(x)
        altered["target_time"]["et"] += 32 * 86400
        changed = model(altered)
    for task in model.tasks:
        torch.testing.assert_close(baseline[task], masked[task])
        torch.testing.assert_close(baseline[task][:1], alone[task], atol=3e-6, rtol=1e-5)
    torch.testing.assert_close(baseline["water"], changed["water"])
    assert not torch.allclose(baseline["et"], changed["et"])
    restored = MODELS.build(config(tmp_path, True).model).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    with torch.no_grad():
        for task, value in restored(x).items():
            torch.testing.assert_close(value, baseline[task])


def test_land_constraint_time_fallback_is_explicit():
    coupling = AlignedLandCoupling()
    target = samples()
    target["water"][:, :, :, :2] = 1
    target["sm_land_time_aligned"] = torch.tensor([True, False])
    prediction = torch.arange(16).reshape(1, 1, 4, 4).float().repeat(2, 1, 1, 1)
    outputs = dict(water=target["water"], soil_moisture=prediction)
    support = coupling.prepare(outputs, target, supervision=True)
    projected = support.project(prediction).flatten()
    torch.testing.assert_close(projected[0], prediction[0, :, :, 2:].mean())
    torch.testing.assert_close(projected[1], prediction[1].mean())
    assert support.valid.all()


def test_real_loss_optimizer_freezing_and_predict(tmp_path):
    cfg = config(tmp_path, True)
    model = MODELS.build(cfg.model).train()
    optimizer = build_optim_wrapper(
        model, dict(type="OptimWrapper", optimizer=dict(type="AdamW", lr=0.001))
    )
    x = raw_inputs(model.tasks)
    x["target_time"]["soil_moisture"][1] += 86400
    y = samples()
    before = model.encoder.backbone.model.blocks[0].attn.q.base.weight.detach().clone()
    losses = model(x, y, mode="loss")
    total = sum(v for k, v in losses.items() if k.startswith("loss_"))
    optimizer.update_params(total)
    torch.testing.assert_close(before, model.encoder.backbone.model.blocks[0].attn.q.base.weight)
    assert all(torch.isfinite(v).all() for v in losses.values())
    model.eval()
    with torch.no_grad():
        p = model(x, y, mode="predict")
        inference = model(x, mode="predict")
    assert p[0]["sm_land_time_aligned"] and not p[1]["sm_land_time_aligned"]
    assert not inference[1]["sm_land_time_aligned"]
    cfg.model.encoder.update(train_patch_embedding=False, train_layer_norm=False, private_rank=0)
    shared = MODELS.build(cfg.model)
    assert not any(
        p.requires_grad for p in shared.encoder.backbone.model.patch_embeddings.parameters()
    )
    assert not any(
        p.requires_grad
        for m in shared.encoder.modules()
        if isinstance(m, torch.nn.LayerNorm)
        for p in m.parameters()
    )
    assert all(not m.private for m in shared.encoder.modules() if isinstance(m, MTLoRALinear))


def test_config_selects_current_five_slot_product_contract():
    cfg = Config.fromfile("configs/hydrostate_mtlora.py")
    configure_data_contract(cfg)
    assert cfg.model.encoder.out_indices == [3, 7, 11]
    assert cfg.model.type == "MTLoRAHydroStateModel"
    assert cfg.input_profile == "landsat_l1_toa_native_mean_v2"
    assert cfg.train_dataloader.dataset.type == "RetrievalDataset"
    assert len(cfg.tasks) == 7
    assert cfg.tasks.soil_moisture.support == "tile_mean_weak"
    assert cfg.optim_wrapper.optimizer.lr == 1e-4
    assert "constructor" not in cfg.optim_wrapper
    assert (
        Config.fromfile("configs/experiments/legacy/hydrostate_multilevel.py").model.type
        == "MultiLevelHydroStateModel"
    )


@pytest.mark.parametrize("checkpoint", [False, True])
def test_full_finetuning_preserves_features_and_updates_backbone(tmp_path, checkpoint):
    cfg = config(tmp_path, checkpoint)
    cfg.model.encoder.update(private_rank=0)
    shared = MODELS.build(cfg.model).eval()
    cfg.model.encoder.update(finetuning="full", shared_rank=0, private_rank=0)
    full = MODELS.build(cfg.model).train()
    core = full.encoder.backbone.model
    assert not full.encoder.backbone.training
    assert not any(isinstance(m, (MTLoRALinear, LowRankUpdate)) for m in full.modules())
    assert not any(p.requires_grad for p in core.project_and_aggregate.parameters())
    assert core.blocks[0].attn.q.weight.requires_grad
    assert core.blocks[-1].mlp.fc2.weight.requires_grad
    x = raw_inputs(full.tasks)
    with torch.no_grad():
        reference, actual = shared.encoder(x), full.encoder(x)
    for task in full.tasks:
        for a, b in zip(reference[task], actual[task]):
            for key in ("features", "valid", "time"):
                torch.testing.assert_close(a[key], b[key])

    options = dict(Config.fromfile("configs/hydrostate_full.py").optim_wrapper)
    options.update(type="OptimWrapper", accumulative_counts=1)
    options.pop("dtype")
    options["paramwise_cfg"] = dict(options["paramwise_cfg"], num_layers=6)
    optimizer = build_optim_wrapper(full, options)
    rates = {id(p): group["lr"] for group in optimizer.optimizer.param_groups
             for p in group["params"]}
    assert set(rates) == {id(p) for p in full.parameters() if p.requires_grad}
    assert rates[id(core.blocks[0].attn.q.weight)] == pytest.approx(1e-4 * .65**6)
    assert rates[id(core.blocks[-1].mlp.fc2.weight)] == pytest.approx(6.5e-5)
    assert rates[id(full.decoders["water"].compress.weight)] == 1e-4
    before = core.blocks[-1].mlp.fc2.weight.detach().clone()
    losses = full(x, samples(), mode="loss")
    total = sum(value for key, value in losses.items() if key.startswith("loss_"))
    optimizer.backward(total)
    assert core.blocks[0].attn.q.weight.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in full.parameters() if p.grad is not None)
    optimizer.step()
    assert not torch.equal(before, core.blocks[-1].mlp.fc2.weight)
    restored = MODELS.build(cfg.model)
    restored.load_state_dict(full.state_dict(), strict=True)


def test_full_configuration_reuses_current_contract_and_multilevel_optimizer():
    full = Config.fromfile("configs/hydrostate_full.py")
    shared = Config.fromfile("configs/hydrostate_mtlora_shared.py")
    legacy = Config.fromfile(
        "configs/experiments/legacy/hydrostate_ecosystem_multilevel_temporal_hpc.py"
    )
    for cfg in (full, shared):
        configure_data_contract(cfg)
    for key in ("tasks", "train_dataloader", "val_dataloader", "test_dataloader",
                "val_evaluator", "test_evaluator", "train_cfg", "randomness"):
        assert full[key] == shared[key]
    for key in full.model:
        if key != "encoder":
            assert full.model[key] == shared.model[key]
    assert full.model.encoder.shared_rank == full.model.encoder.private_rank == 0
    assert full.model.encoder.finetuning == "full"
    assert full.optim_wrapper.paramwise_cfg == legacy.optim_wrapper.paramwise_cfg
    for key in ("lr", "weight_decay"):
        assert full.optim_wrapper.optimizer[key] == legacy.optim_wrapper.optimizer[key]
    assert full.optim_wrapper.clip_grad == legacy.optim_wrapper.clip_grad
    assert full.optim_wrapper.accumulative_counts == legacy.optim_wrapper.accumulative_counts == 8
    assert full.param_scheduler == legacy.param_scheduler


def test_full_mode_rejects_low_rank_adapters_before_loading_pretrained(tmp_path):
    cfg = config(tmp_path)
    cfg.model.encoder.finetuning = "full"
    with pytest.raises(ValueError, match="shared_rank=private_rank=0"):
        MODELS.build(cfg.model)
