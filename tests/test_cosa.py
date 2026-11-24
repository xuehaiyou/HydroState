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


@pytest.mark.parametrize("enabled", [False, True, "ditask_v1", "d2moe_v1", "ditask", "d2moe",
                                     "nexus_d2moe", "faar", "rutal",
                                     "cgc", "et_only", "recurrent_cgc", "background", "mlore",
                                     "ple", "mlore_complete"])
def test_cosa_mixed_product_station_supervision_reaches_existing_heads(tmp_path, enabled):
    stage_methods = ("ditask", "d2moe", "nexus_d2moe", "faar", "rutal")
    cfg = config(tmp_path, enabled, checkpoint=True)
    if enabled in ("ditask_v1", "d2moe_v1"):
        cfg.model.type = "FoundationStageModel"
        cfg.model.encoder.update(type="FoundationStageEncoder", method=enabled.removesuffix("_v1"))
        cfg.model.router.enabled = False
    if enabled in stage_methods:
        cfg.model.type = "FoundationStageModelV2"
        if enabled == "nexus_d2moe":
            cfg.model.type = "NexusD2MoE"
        if enabled in ("d2moe", "nexus_d2moe"):
            cfg.model.decoder.type = "D2MoEStateFusion"
        elif enabled in ("rutal", "faar"):
            cfg.model.decoder.type = {"rutal": "RUTaLDecoder", "faar": "FAARDecoder"}[enabled]
        else:
            pytest.importorskip("difw")
            cfg.model.decoder.type = "DiTASKStateFusion"
        cfg.model.encoder.update(type="FoundationStageEncoderV2",
                                 method=enabled.removeprefix("nexus_"),
                                 num_levels=4, out_indices=[0, 1, 3, 5])
        cfg.model.decoder.num_levels = 4
        if enabled == "rutal":
            cfg.model.decoder.update(num_levels=5, ladder_heads=2)
        cfg.model.manual_cover_weight = 2.0
        cfg.model.router.enabled = False
    if enabled == "et_only":
        cfg.model.type = "ETOnlyCGC"
    if enabled in ("cgc", "et_only"):
        cfg.model.router = dict(type="CGCRouter", num_shared=2)
    if enabled == "background":
        cfg.model.type = "BackgroundResidualModel"
        cfg.model.router = dict(center=True)
    if enabled == "recurrent_cgc":
        cfg.model.router = dict(type="RecurrentCGCRouter", spatial_gates=["sm"], depth=3)
    if enabled == "mlore":
        cfg.model.router = dict(type="MLoRERouter", grid_size=3,
                                ranks=(2, 4), private_rank=2, topk=1)
    if enabled in ("ple", "mlore_complete"):
        cfg.model.type = "GroupedExpertHydroStateModel"
        cfg.model.branches = None
        cfg.model.router = None
        if enabled == "ple":
            cfg.model.decoder = dict(type="PLEDecoder", groups=["coverage", "sm", "et"],
                temporal=dict(in_channels=32, channels=16, quality_channels=4),
                widths=(16, 8), tower_channels=8)
        else:
            cfg.model.official_heads = True
            cfg.model.encoder.out_indices = [0, 1, 3, 5]
            cfg.model.encoder.num_levels = 4
            cfg.model.decoder = dict(type="CompleteMLoREDecoder", groups=["coverage", "sm", "et"],
                in_channels=32, channels=8, grid_size=3, ranks=(2, 4), private_rank=2,
                topk=1, quality_channels=4)
    cfg.model.site_supervision = dict(
        statistics=dict(version="site_training_statistics_v1", split="train", ddof=0,
                        training_fingerprint="a" * 64, observations={
                            "site_sm": dict(count=10, mean=.2, std=.1),
                            "site_et": dict(count=10, mean=2., std=1.),
                        }), loss_weights=dict(site_sm=2., site_et=2.))
    # HPC config.dump sorts mapping keys; group identity must not depend on ordering.
    config_file = tmp_path / "roundtrip.py"
    cfg.dump(str(config_file))
    cfg = Config.fromfile(str(config_file))
    model = MODELS.build(cfg.model).train()
    if enabled in ("ditask", "d2moe", "nexus_d2moe"):
        from hydrostate.models.cosa.ditask import AdapterDropout
        drops = [mod for mod in model.encoder.backbone.modules() if isinstance(mod, AdapterDropout)]
        model.eval()
        assert drops and not any(d.training for d in drops)
        model.train()
        assert all(d.training for d in drops) and not model.encoder.backbone.training
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
    if enabled in stage_methods:
        samples = raw["data_samples"]
        for task in ("water", "snow_ice", "land"):
            samples["manual_" + task] = torch.full_like(samples["water"], float(task == "water"))
            mask = torch.zeros_like(samples["water"], dtype=torch.bool)
            mask[0, :, 2, 2] = True
            samples["manual_" + task + "_valid"] = mask
    data = model.data_preprocessor(raw, training=True)
    losses = model(**data, mode="loss")
    if enabled == "et_only":
        assert {k for k in losses if k.startswith("loss_")} == {
            "loss_et_water", "loss_et_ice", "loss_et_land", "loss_site_et"}
        losses["loss_site_et"].backward()
        assert all(p.grad is None for p in model.heads["soil_moisture"].parameters())
        assert any(p.grad.abs().sum() > 0 for p in model.heads["et_land"].parameters())
        return
    assert losses["valid_count_site_sm"] == losses["valid_count_site_et"] == 1
    assert losses["valid_count_soil_moisture"] == 1
    site_loss = losses["loss_site_sm"] + losses["loss_site_et"]
    if enabled in stage_methods:
        assert losses["loss_manual_cover"] > 0
        (site_loss + losses["loss_manual_cover"] + losses.get("loss_nexus_alignment", 0)).backward()
        if enabled == "nexus_d2moe":
            assert losses["loss_nexus_alignment"] > 0
            assert all(next(m.parameters()).grad.abs().sum() > 0
                       for m in model.alignment.surrogates.values())
        assert model.heads["coverage"].weight.grad.abs().sum() > 0
    else:
        site_loss.backward()
    for task in (("sm", "et") if enabled in ("mlore_complete", *stage_methods) else
                 ("soil_moisture", "et_water", "et_ice", "et_land")):
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


def test_batched_depth_pooling_matches_separate_depths_and_gradients():
    torch.manual_seed(7)
    fusion = CoSAStateFusion(in_channels=8, channels=8, quality_channels=4)
    levels = []
    for _ in range(3):
        value = torch.randn(2, 3, 2, 2, 8, requires_grad=True)
        valid = torch.rand(2, 3, 2, 2) > .3
        times = torch.tensor([[1600000000, 1600691200, 1601382400]]).expand(2, -1)
        levels.append((value, valid, times))
    quality = torch.randn(2, 8, 2, 2, requires_grad=True)
    valid = torch.stack([item[1].any(1) for item in levels]).any(0)[:, None]
    bounds = torch.tensor([[1600691200, 1601382400]]).expand(2, -1)
    separate = [fusion.pooler(value, mask, times, bounds, quality)
                for value, mask, times in levels]
    expected = fusion.spatial(torch.nn.functional.gelu(
        fusion.compress(torch.cat(separate, 1)) + quality), valid)
    actual, _ = fusion((levels, quality, valid), bounds)
    torch.testing.assert_close(actual, expected)
    inputs = [item[0] for item in levels] + [quality]
    left = torch.autograd.grad(expected.sum(), inputs, retain_graph=True)
    right = torch.autograd.grad(actual.sum(), inputs)
    for a, b in zip(left, right):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)


def test_vectorized_router_matches_directed_loop_and_gradients():
    torch.manual_seed(8)
    router = StateRouter(('coverage', 'sm', 'et'), state_channels=8, channels=4,
                         task_embedding_dim=3, routing_dim=4, residual_init=.3,
                         sources=dict(coverage=('et',), sm=('coverage', 'et'), et=()))
    state = torch.randn(2, 8, 3, 3, requires_grad=True)
    features = {name: torch.randn(2, 4, 3, 3, requires_grad=True) for name in router.experts}
    valid = torch.rand(2, 1, 3, 3) > .2
    embeddings = {name: router.task_embedding.weight[i][None, :, None, None].expand(2, -1, 3, 3)
                  for i, name in enumerate(router.experts)}
    expected = {}
    for name in router.experts:
        value = features[name]
        query = router.query(torch.cat((state, embeddings[name]), 1))
        for source in router.sources[name]:
            expert = features[source].detach().masked_fill(~valid, 0)
            key = router.key(torch.cat((expert, embeddings[source]), 1))
            gate = ((query.float() * key.float()).sum(1, keepdim=True)
                    * router.scale).sigmoid().masked_fill(~valid, 0)
            value = value + router.gamma[name] * gate * router.value(expert) / len(
                router.sources[name])
        expected[name] = value.masked_fill(~valid, 0)
    actual = router(state, features, valid)
    for name in router.experts:
        torch.testing.assert_close(actual[name], expected[name])
    parameters = [state, *features.values(), *router.parameters()]
    left = torch.autograd.grad(sum(v.sum() for v in expected.values()), parameters,
                               retain_graph=True, allow_unused=True)
    right = torch.autograd.grad(sum(v.sum() for v in actual.values()), parameters,
                                allow_unused=True)
    for a, b in zip(left, right):
        if a is None:
            assert b is None or torch.count_nonzero(b) == 0
        else:
            torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)


def test_real_encoder_uses_broadcast_mask_and_no_default_checkpoint(tmp_path):
    from functools import partial

    from hydrostate.models.olmoearth_encoder import _broadcast_sdpa

    cfg = config(tmp_path)
    assert not cfg.model.encoder.gradient_checkpointing
    model = MODELS.build(cfg.model)
    for block in model.encoder.backbone.model.blocks:
        assert isinstance(block.attn.sdpa, partial)
        assert block.attn.sdpa.func is _broadcast_sdpa
        assert not isinstance(block.forward, partial)


def test_cgc_sm_refinement_preserves_every_non_sm_prediction(tmp_path, tiny):
    cfg = config(tmp_path)
    cfg.model.router = dict(type="CGCRouter", spatial_gates=["sm"])
    cfg.model.type = "CGCSMRefinement"
    model = MODELS.build(cfg.model).train()
    assert not model.encoder.training and not model.decoder.training
    inputs = model.data_preprocessor(batch(), training=True)["inputs"]
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.1)
    before = model(inputs)
    before["soil_moisture"].mean().backward()
    optimizer.step()
    after = model.eval()(inputs)
    assert not torch.equal(after["soil_moisture"], before["soil_moisture"])
    for name in set(before) - {"soil_moisture"}:
        torch.testing.assert_close(after[name], before[name], atol=0, rtol=0)


def test_centering_changes_only_valid_spatial_mean():
    from hydrostate.models.cosa.branches import BackgroundResidual

    torch.manual_seed(42)
    model = BackgroundResidual(['sm', 'et'], 8, 4, center=False)
    x = torch.randn(2, 8, 5, 5)
    valid = torch.ones_like(x[:, :1], dtype=torch.bool)
    valid[0] = False
    valid[1, :, 1, 2] = False
    x.masked_fill_(~valid, float('nan'))
    raw = model(x, valid)
    model.center = True
    centered = model(x, valid)
    for name, output in centered.items():
        assert not output[0].any() and not output[1, :, 1, 2].any()
        background = model.background[name](x[1, :, valid[1, 0]].mean(-1).view(1, 8, 1, 1))
        difference = raw[name][1, :, valid[1, 0]] - output[1, :, valid[1, 0]]
        torch.testing.assert_close(difference, difference.mean(-1, keepdim=True).expand_as(difference))
        torch.testing.assert_close(output[1, :, valid[1, 0]].mean(-1), background.flatten())

    model.scene_mean_response = True
    adjusted = model(x, valid)
    for name in centered:
        shift = (adjusted[name] - centered[name])[1, :, valid[1, 0]]
        assert shift.abs().max() > 0
        torch.testing.assert_close(shift, shift.mean(-1, keepdim=True).expand_as(shift))
        assert not adjusted[name][0].any()


def test_detail_offsets_preserve_base_and_masked_mean():
    from hydrostate.models.cosa.cgc import CGCRouter

    base = CGCRouter(["et"], 8, 4)
    new = CGCRouter(["et"], 8, 4, separate_detail=True)
    assert new.load_state_dict(base.state_dict(), strict=False).missing_keys == ["detail_offsets.et"]
    x = torch.randn(2, 8, 5, 5)
    valid = torch.rand(2, 1, 5, 5) > .3
    features = {"et": torch.randn(2, 4, 5, 5)}
    ref = base(x, features, valid)["et"]
    y = new(x, features, valid)["et"]
    torch.testing.assert_close(ref, y, atol=0, rtol=0)
    y.square().sum().backward()
    assert new.detail_offsets["et"].grad.abs().sum() > 0
    with torch.no_grad():
        new.detail_offsets["et"][0] = 1
    diff = new(x, features, valid)["et"] - ref
    assert diff.abs().max() > .01
    torch.testing.assert_close(diff.sum((-2, -1)), torch.zeros(2, 4), atol=1e-5, rtol=0)
