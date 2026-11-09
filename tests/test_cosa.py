"""CoSA scientific masks, gradient paths, paired initialization and public APIs."""

import copy

import pytest
import torch
from mmengine.config import Config
from mmengine.optim import build_optim_wrapper
from retrieval.test_model import batch, statistics
from test_mtlora import tiny  # noqa: F401 -- shared real tiny OlmoEarth fixture

from hydrostate.cli import configure_data_contract
from hydrostate.models.cosa.routing import StateRouter
from hydrostate.models.cosa.state import CoSAStateFusion
from hydrostate.models.mtlora import MTLoRALinear
from hydrostate.registry import MODELS


def config(tmp_path, enabled=True, checkpoint=False):
    cfg = Config.fromfile("configs/hydrostate_cosa.py" if enabled
                          else "configs/hydrostate_cosa_independent.py")
    configure_data_contract(cfg)
    cfg.model.encoder.update(model_path=str(tmp_path), embedding_channels=32,
                             out_indices=[1, 3, 5], shared_rank=2,
                             gradient_checkpointing=checkpoint)
    cfg.model.decoder.update(in_channels=32, channels=16, quality_channels=4)
    cfg.model.branches.channels = 8
    cfg.model.router.update(task_embedding_dim=4, routing_dim=4)
    for head in cfg.model.heads.values():
        head.in_channels = 8
    cfg.model.target_statistics = statistics()
    return cfg


def test_configs_only_switch_interaction_and_keep_supervision_budget():
    independent = Config.fromfile("configs/hydrostate_cosa_independent.py")
    joint = Config.fromfile("configs/hydrostate_cosa.py")
    original = Config.fromfile("configs/hydrostate_mtlora_shared.py")
    for cfg in (independent, joint, original):
        configure_data_contract(cfg)
    a, b = copy.deepcopy(independent.model), copy.deepcopy(joint.model)
    a.router.enabled = True
    assert a == b
    for key in ("train_dataloader", "val_dataloader", "test_dataloader", "param_scheduler",
                "optim_wrapper", "train_cfg", "tasks", "data_contract"):
        assert independent[key] == joint[key] == original[key]
    assert independent.load_from is joint.load_from is None
    assert independent.randomness == joint.randomness
    assert joint.model.encoder.use_landsat_toa_adapter is False
    assert original.model.encoder.get("use_landsat_toa_adapter", True)


@pytest.mark.parametrize("enabled", [False, True])
def test_cosa_mixed_product_station_supervision_reaches_existing_heads(tmp_path, enabled):
    cfg = config(tmp_path, enabled, checkpoint=True)
    cfg.model.site_supervision = dict(
        statistics=dict(version="site_training_statistics_v1", split="train", ddof=0,
                        training_fingerprint="a" * 64, observations={
                            "site_sm": dict(count=10, mean=.2, std=.1),
                            "site_et": dict(count=10, mean=2., std=1.),
                        }), loss_weights=dict(site_sm=2., site_et=2.))
    model = MODELS.build(cfg.model).train()
    raw = batch()
    for name, value in raw["data_samples"].items():
        if name.endswith("valid") and torch.is_tensor(value):
            value[1].zero_()  # Second record has station supervision only.
    for name, value in (("site_sm", .2), ("site_et", 2.)):
        raw["data_samples"][f"{name}_native_value"] = torch.full((2, 1), value)
        raw["data_samples"][f"{name}_native_valid"] = torch.tensor([[False], [True]])
        weights = torch.zeros(2, 1, 4, 4)
        weights[1, 0, 2, 2] = 1
        raw["data_samples"][f"{name}_native_weights"] = weights
    data = model.data_preprocessor(raw, training=True)
    losses = model(**data, mode="loss")
    assert losses["valid_count_site_sm"] == losses["valid_count_site_et"] == 1
    assert losses["valid_count_soil_moisture"] == 1
    (losses["loss_site_sm"] + losses["loss_site_et"]).backward()
    for task in ("soil_moisture", "et_water", "et_ice", "et_land"):
        assert any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in model.heads[task].parameters())
    predictions = model.predict(**data)
    assert predictions[1]["valid_site_sm"].all()
    assert not predictions[0]["valid_site_sm"].any()


@pytest.mark.parametrize("checkpoint", [False, True])
def test_paired_initialization_single_pass_trainability_and_zero_transfer(tmp_path, checkpoint):
    models = []
    states = []
    for enabled in (False, True):
        torch.manual_seed(3407)
        models.append(MODELS.build(config(tmp_path, enabled, checkpoint).model).train())
        states.append(torch.get_rng_state())
    left, right = models
    assert torch.equal(*states)
    for name, value in left.state_dict().items():
        torch.testing.assert_close(value, right.state_dict()[name], atol=0, rtol=0)
    assert not any(p.requires_grad for p in left.router.parameters())
    assert right.encoder.landsat_toa_adapter is None
    assert right.encoder.landsat_l2_adapter is None
    assert not right.encoder.backbone.training
    core = right.encoder.backbone.model
    assert all(p.requires_grad for p in core.patch_embeddings.parameters())
    assert core.blocks[0].norm1.weight.requires_grad
    assert not any(p.requires_grad for p in core.project_and_aggregate.parameters())
    for module in right.encoder.modules():
        if isinstance(module, MTLoRALinear):
            assert not module.base.weight.requires_grad
            assert not module.private
    data = right.data_preprocessor(batch(), training=True)
    calls = []
    handles = [block.attn.register_forward_hook(lambda *args: calls.append(1))
               for block in core.blocks]
    output = right(data["inputs"])
    for handle in handles:
        handle.remove()
    assert len(calls) == 6
    for task, reference in left(data["inputs"]).items():
        torch.testing.assert_close(reference, output[task], atol=0, rtol=0)
    output["soil_moisture"].mean().backward()
    assert right.router.gamma["sm"].grad.abs() > 0
    assert core.blocks[0].attn.q.shared.b.weight.grad.abs().sum() > 0
    assert core.blocks[0].norm1.weight.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in core.patch_embeddings.parameters())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in right.parameters())


def test_router_detaches_both_source_paths_and_learns_after_zero_start():
    torch.manual_seed(13)
    router = StateRouter(("coverage", "sm", "et"), state_channels=8, channels=4,
                         task_embedding_dim=3, routing_dim=4)
    state = torch.randn(2, 8, 3, 3, requires_grad=True)
    features = {name: torch.randn(2, 4, 3, 3, requires_grad=True) for name in router.experts}
    valid = torch.ones(2, 1, 3, 3, dtype=torch.bool)
    optim = torch.optim.SGD(router.parameters(), lr=.1)
    router(state, features, valid)["sm"].square().mean().backward()
    assert router.gamma["sm"].grad.abs() > 0
    assert router.key.weight.grad.abs().sum() == 0
    optim.step()
    optim.zero_grad(set_to_none=True)
    for feature in features.values():
        feature.grad = None
    router(state, features, valid)["sm"].square().mean().backward()
    assert features["coverage"].grad is None and features["et"].grad is None
    assert features["sm"].grad.abs().sum() > 0
    assert state.grad.abs().sum() > 0
    for layer in (router.query, router.key, router.value):
        assert layer.weight.grad.abs().sum() > 0
    router.sources["sm"] = ()
    torch.testing.assert_close(router(state, features, valid)["sm"], features["sm"])
    with pytest.raises(ValueError, match="sources"):
        StateRouter(("a", "b"), sources=dict(a=["a"], b=[]))


def test_state_ignores_invalid_values_and_preserves_empty_support_and_interval():
    torch.manual_seed(4)
    fusion = CoSAStateFusion(in_channels=8, channels=8, quality_channels=4)
    inputs = batch()["inputs"]
    bounds = next(iter(inputs["target_time"].values()))
    times = inputs["observation_time"][:, :3, 0]
    mask = torch.ones(2, 3, 3, 3, dtype=torch.bool)
    mask[:, :, 0, 0] = False
    mask[:, 1:, 1, 1] = False  # Only one observation still responds to the query.
    levels = [dict(features=torch.randn(2, 3, 8, 3, 3), valid=mask.clone(), time=times)
              for _ in range(3)]
    changed = copy.deepcopy(levels)
    for level in changed:
        level["features"].masked_fill_(~mask[:, :, None], float("nan"))
    reference, valid = fusion(fusion.prepare(levels, inputs), bounds)
    actual, _ = fusion(fusion.prepare(changed, inputs), bounds)
    torch.testing.assert_close(reference, actual)
    assert not valid[:, :, 0, 0].any()
    assert torch.equal(reference[:, :, 0, 0], torch.zeros_like(reference[:, :, 0, 0]))
    later, _ = fusion(fusion.prepare(levels, inputs), bounds + 86400 * 30)
    assert not torch.allclose(reference[:, :, 1, 1], later[:, :, 1, 1])
    for level in levels:
        level["valid"].zero_()
    empty, valid = fusion(fusion.prepare(levels, inputs), bounds)
    assert not valid.any() and torch.isfinite(empty).all() and empty.count_nonzero() == 0


@pytest.mark.parametrize("enabled", [False, True])
def test_loss_update_predict_missing_task_and_checkpoint(tmp_path, enabled):
    cfg = config(tmp_path, enabled, checkpoint=True)
    model = MODELS.build(cfg.model).train()
    optimizer = build_optim_wrapper(
        model, dict(type="OptimWrapper", optimizer=dict(type="AdamW", lr=.001)))
    data = model.data_preprocessor(batch(), training=True)
    frozen = model.encoder.backbone.model.blocks[0].attn.q.base.weight.detach().clone()
    for _ in range(2):
        loss = model(**data, mode="loss")
        assert all(torch.isfinite(v).all() for v in loss.values())
        optimizer.update_params(sum(v for k, v in loss.items() if k.startswith("loss_")))
    torch.testing.assert_close(frozen, model.encoder.backbone.model.blocks[0].attn.q.base.weight)
    if enabled:
        assert model.router.gamma["sm"].abs() > 0
    model.eval()
    output = model(data["inputs"])
    assert all(v.shape == (2, 1, 4, 4) for v in output.values())
    torch.testing.assert_close(sum(output[t] for t in ("water", "snow_ice", "land")),
                               torch.ones(2, 1, 4, 4))
    assert all((output[t] >= 0).all() for t in ("et_water", "et_ice", "et_land"))
    loss = model(**data, mode="loss")
    data["data_samples"]["soil_moisture_valid"].zero_()
    missing = model(**data, mode="loss")
    assert missing["valid_count_soil_moisture"] == 0
    torch.testing.assert_close(missing["loss_et_land"], loss["loss_et_land"])
    predictions = model(**data, mode="predict")
    for i, pred in enumerate(predictions):
        torch.testing.assert_close(pred["supervised_soil_moisture"].flatten(),
                                   output["soil_moisture"][i].mean().reshape(1))
        torch.testing.assert_close(pred["et_total"], sum(
            output[task][i] * output[cover][i] for task, cover in
            (("et_water", "water"), ("et_ice", "snow_ice"), ("et_land", "land"))))
    for sensor in ("s1", "s2", "landsat"):
        data["inputs"][f"{sensor}_valid"][:, :, :, :3, :3] = False
    predictions = model(data["inputs"], mode="predict")
    assert not predictions[0]["et_total_prediction_valid"][0, 0, 0]
    assert torch.isnan(predictions[0]["et_total"][0, 0, 0])
    path = tmp_path / "cosa.pth"
    torch.save(model.state_dict(), path)
    restored = MODELS.build(cfg.model).eval()
    restored.load_state_dict(torch.load(path, weights_only=True), strict=True)
    for task, value in model(data["inputs"]).items():
        torch.testing.assert_close(value, restored(data["inputs"])[task])


def test_query_group_reuses_state_and_never_uses_sources_at_another_interval(tmp_path):
    cfg = config(tmp_path)
    cfg.model.query_groups = [["water", "snow_ice", "land"], ["soil_moisture"],
                              ["et_water", "et_ice", "et_land"]]
    model = MODELS.build(cfg.model).eval()
    for gamma in model.router.gamma.values():
        gamma.data.fill_(.2)
    data = model.data_preprocessor(batch())["inputs"]
    reference = model(data)
    for task in cfg.model.expert_groups.et:
        data["target_time"][task] += 32 * 86400
    changed = model(data)
    for task in ("water", "snow_ice", "land", "soil_moisture"):
        torch.testing.assert_close(reference[task], changed[task])
    assert not torch.allclose(reference["et_land"], changed["et_land"])
    shared = MODELS.build(config(tmp_path).model)
    with pytest.raises(RuntimeError, match="matching target intervals"):
        shared(data)


def test_existing_regional_loader_and_batched_predict_accept_cosa(tmp_path):
    import json

    from hydrostate.data.retrieval.contract import PROFILE
    from hydrostate.inference.regional import load_model, predict_records

    cfg = config(tmp_path)
    model = MODELS.build(cfg.model).eval()
    config_path = tmp_path / "model.py"
    cfg.dump(str(config_path))
    checkpoint = tmp_path / "checkpoint.pth"
    torch.save(dict(state_dict=model.state_dict()), checkpoint)
    for name in ("config.json", "weights.pth"):
        (tmp_path / name).write_text("tiny-loader-fixture")
    stats = statistics() | dict(data_contract=dict(input_profile=PROFILE))
    stats_path = tmp_path / "statistics.json"
    stats_path.write_text(json.dumps(stats))
    loaded, provenance = load_model(config_path, checkpoint, str(tmp_path), stats_path, "cpu")
    inputs = batch()["inputs"]
    records = [dict(inputs={
        key: ({task: value[i] for task, value in item.items()} if isinstance(item, dict)
              else item[i]) for key, item in inputs.items()}) for i in range(2)]
    predictions = predict_records(loaded, records, "cpu")
    assert len(predictions) == 2
    assert predictions[0]["land"].shape == (1, 4, 4)
    assert provenance["model_config"]["type"] == "CoSAHydroStateModel"
    assert len(provenance["checkpoint_sha256"]) == 64
