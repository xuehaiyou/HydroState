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
from hydrostate.models.mtlora import AlignedLandCoupling, MTLoRALinear
from hydrostate.models.multilevel_temporal_model import MultiLevelTemporalOlmoEarthEncoder
from hydrostate.registry import MODELS
from hydrostate.tasks import task_config


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
    cfg.model.target_statistics = moments(task_config())
    return cfg


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
    encoder = MODELS.build(dict(cfg.model.encoder, tasks=task_config())).eval()
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
        altered["target_time"]["et"] += 86400
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


def test_config_selects_new_architecture_without_altering_legacy():
    cfg = Config.fromfile("configs/hydrostate_mtlora.py")
    configure_data_contract(cfg)
    assert cfg.model.encoder.out_indices == [3, 7, 11]
    assert cfg.model.type == "MTLoRAHydroStateModel"
    assert cfg.input_profile == "olmoearth_l1_v1" and not cfg.input_packs
    assert cfg.train_dataloader.dataset.target_time_queries
    assert cfg.optim_wrapper.optimizer.lr == 1e-4
    assert "constructor" not in cfg.optim_wrapper
    assert (
        Config.fromfile("configs/hydrostate_multilevel.py").model.type
        == "MultiLevelHydroStateModel"
    )
