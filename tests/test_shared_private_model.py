"""Parameter routing, transfer and MMEngine compatibility on real tiny OlmoEarth."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from mmengine.config import Config
from mmengine.optim import build_optim_wrapper
from mmengine.registry import init_default_scope

from hydrostate.cli import configure_data_contract
from hydrostate.models.shared_private_lora import SharedPrivateLinear, TaskContext, task_context
from hydrostate.models.shared_private_transfer import load_adapter_bundle, save_adapter_bundle
from hydrostate.normalization import CONFIGURED_SUPPORT_VERSION
from hydrostate.registry import MODELS
from hydrostate.tasks import task_config


def install_tiny_backbone(monkeypatch):
    from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.nn.flexi_vit import EncoderConfig
    from rslearn.models.olmoearth_pretrain import model as wrapper

    def load(*args, **kwargs):
        # Every instance represents the same pretrained checkpoint.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(51)
            core = EncoderConfig(
                supported_modality_names=["sentinel1", "sentinel2_l2a", "landsat"],
                embedding_size=32, num_heads=4, depth=3, drop_path=0,
                num_register_tokens=0, use_flash_attn=False, use_linear_patch_embed=True,
                position_encoding="rope_3d_mixed",
            ).build()
        return SimpleNamespace(encoder=core)

    monkeypatch.setattr(wrapper, "load_model_from_path", load)
    init_default_scope("hydrostate")


@pytest.fixture(autouse=True)
def tiny_olmoearth(monkeypatch):
    install_tiny_backbone(monkeypatch)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def make_config(tmp_path, variant="", checkpoint=True):
    cfg = Config.fromfile(f"configs/experiments/legacy/hydrostate_shared_private{variant}.py")
    configure_data_contract(cfg)
    cfg.model.adapter_init = None
    cfg.model.encoder.update(model_path=str(tmp_path), embedding_channels=32,
                             out_indices=[0, 1, 2], gradient_checkpointing=checkpoint,
                             shared_rank=2, private_rank=2)
    cfg.model.decoder.update(in_channels=32, channels=[8, 8, 4])
    cfg.model.query_time.query_channels = 8
    for head in cfg.model.heads.values():
        head.update(in_channels=4, hidden_channels=4)
    cfg.model.target_statistics = dict(
        version=2, split="train", ddof=0, support_version=CONFIGURED_SUPPORT_VERSION,
        training_fingerprint="0" * 64, task_specs=task_config(cfg.model.tasks),
        tasks={t: dict(count=3, mean=float(i), std=1. + i)
               for i, t in enumerate(cfg.model.tasks)},
        data_contract=dict(input_profile="landsat_l2_sr_v1"),
    )
    return cfg


def inputs(tasks):
    result = dict(
        sensor_valid=torch.ones(2, 2, 3, dtype=torch.bool),
        observation_time=torch.tensor([[[1600000000] * 3, [1600086400] * 3]] * 2),
        target_time={t: torch.full((2, 2), 1600007200) for t in tasks},
    )
    for key, channels in (("s1", 2), ("s2", 12), ("landsat_l2", 7)):
        result[key] = torch.randn(2, 2, channels, 12, 12)
        result[key + "_valid"] = torch.ones(2, 2, channels, 12, 12, dtype=torch.bool)
    result["landsat_l2_time"] = result["observation_time"][:, :, 2].clone()
    return result


def samples(extra=()):
    support = torch.ones(2, 1, 4, 4, dtype=torch.bool)
    result = dict(
        prediction_valid=support,
        water=torch.full((2, 1, 4, 4), .3), water_valid=support.clone(),
        soil_moisture=torch.full((2, 1, 1, 1), .2),
        soil_moisture_valid=torch.ones(2, 1, 1, 1, dtype=torch.bool),
        et_native_weights=torch.full((2, 1, 4, 4), 1 / 16),
        et_native_value=torch.ones(2, 1), et_native_valid=torch.ones(2, 1, dtype=torch.bool),
    )
    if "lai" in extra:
        result.update(lai_native_weights=result["et_native_weights"].clone(),
                      lai_native_value=torch.full((2, 1), 2.),
                      lai_native_valid=torch.ones(2, 1, dtype=torch.bool))
    return result


def optimizer(model):
    return build_optim_wrapper(model, dict(
        type="OptimWrapper", constructor="SharedPrivateOptimWrapperConstructor",
        optimizer=dict(type="AdamW", lr=.002, weight_decay=.1)))


def step(model, x, y, optim):
    optim.zero_grad(set_to_none=True)
    losses = model(x, y, mode="loss")
    total = sum(value for name, value in losses.items() if name.startswith("loss_"))
    optim.update_params(total)
    return losses


def test_formula_initialization_and_task_gradients():
    layer = SharedPrivateLinear(torch.nn.Linear(4, 3), 0, 2, dict(water=2, et=2))
    x = torch.randn(2, 4)
    gates = torch.nn.Parameter(torch.zeros(1))
    with task_context(TaskContext("water", gates)):
        torch.testing.assert_close(layer(x), layer.base(x), atol=0, rtol=0)
        torch.nn.init.normal_(layer.shared.b.weight)
        torch.nn.init.normal_(layer.private["water"].b.weight)
        expected = layer.base(x) + .5 * layer.shared(x) + layer.private["water"](x)
        torch.testing.assert_close(layer(x), expected)
        layer(x).square().sum().backward()
    assert gates.grad.abs().sum() > 0
    assert layer.shared.b.weight.grad.abs().sum() > 0
    assert layer.private["water"].a.weight.grad.abs().sum() > 0
    assert layer.private["et"].b.weight.grad is None
    assert layer.base.weight.grad is None
    with pytest.raises(RuntimeError, match="explicit"):
        layer(x)


def test_checkpoint_recomputation_and_deepcopy_keep_task_gradients(tmp_path):
    cfg = make_config(tmp_path)
    checkpointed = MODELS.build(cfg.model).train()
    for module in checkpointed.modules():
        if isinstance(module, SharedPrivateLinear):
            torch.nn.init.normal_(module.shared.b.weight, std=.05)
            for update in module.private.values():
                torch.nn.init.normal_(update.b.weight, std=.05)
    cfg.model.encoder.gradient_checkpointing = False
    reference = MODELS.build(cfg.model).train()
    reference.load_state_dict(checkpointed.state_dict(), strict=True)
    cloned = copy.deepcopy(checkpointed)
    x, y = inputs(checkpointed.tasks), samples()
    for model in (checkpointed, reference, cloned):
        losses = model(x, y, mode="loss")
        sum(v for k, v in losses.items() if k.startswith("loss_")).backward()
    for name, parameter in checkpointed.named_parameters():
        expected = dict(reference.named_parameters())[name]
        copied = dict(cloned.named_parameters())[name]
        if parameter.grad is None:
            assert expected.grad is None and copied.grad is None, name
        else:
            assert torch.isfinite(parameter.grad).all(), name
            torch.testing.assert_close(parameter.grad, expected.grad, msg=name)
            torch.testing.assert_close(parameter.grad, copied.grad, msg=name)
        if checkpointed.parameter_owner(name) is None:
            assert not parameter.requires_grad and parameter.grad is None, name
    assert checkpointed.encoder.gates["water"].grad.abs().sum() > 0
    assert checkpointed.encoder.landsat_l2_adapter.adapters[
        "soil_moisture"].projection.weight.grad.abs().sum() > 0


def test_specialization_and_missing_labels_preserve_other_parameters(tmp_path):
    model = MODELS.build(make_config(tmp_path).model).train()
    x, y = inputs(model.tasks), samples()
    optim = optimizer(model)
    step(model, x, y, optim)
    # Previously active AdamW momentum must not move a task absent in this batch.
    missing = copy.deepcopy(y)
    missing["water_valid"].zero_()
    before = copy.deepcopy(model.state_dict())
    losses = step(model, x, missing, optim)
    assert "loss_water" not in losses
    for name, value in model.state_dict().items():
        if model.parameter_owner(name) == "water":
            assert torch.equal(value, before[name]), name
    model.set_training_tasks(["soil_moisture"], train_shared=False)
    optim = optimizer(model)
    members = {id(p) for group in optim.optimizer.param_groups for p in group["params"]}
    assert members == {id(p) for p in model.parameters() if p.requires_grad}
    assert all(model.parameter_owner(n) == "soil_moisture"
               for n, p in model.named_parameters() if p.requires_grad)
    model.eval()
    with torch.no_grad():
        old_predictions = model(x)
    model.train()
    before = copy.deepcopy(model.state_dict())
    step(model, x, y, optim)
    assert any(not torch.equal(v, before[n]) for n, v in model.state_dict().items()
               if model.parameter_owner(n) == "soil_moisture")
    for name, value in model.state_dict().items():
        if model.parameter_owner(name) != "soil_moisture":
            assert torch.equal(value, before[name]), name
    model.eval()
    with torch.no_grad():
        after = model(x)
    for task in ("water", "et"):
        torch.testing.assert_close(old_predictions[task], after[task], atol=0, rtol=0)
    empty = copy.deepcopy(y)
    empty["soil_moisture_valid"].zero_()
    before = copy.deepcopy(model.state_dict())
    assert "loss_empty" in step(model, x, empty, optim)
    assert all(torch.equal(v, before[n]) for n, v in model.state_dict().items())


def test_transfer_new_lai_and_compose_private_branches(tmp_path):
    original = MODELS.build(make_config(tmp_path).model)
    x, y = inputs(original.tasks), samples()
    step(original, x, y, optimizer(original))
    bundle = tmp_path / "joint.pth"
    save_adapter_bundle(original, bundle)
    with pytest.raises(FileExistsError):
        save_adapter_bundle(original, bundle)
    incremental = MODELS.build(make_config(tmp_path, "_lai").model)
    load_adapter_bundle(incremental, bundle)
    incremental.eval()
    original.eval()
    expanded = dict(x, target_time=dict(x["target_time"], lai=x["target_time"]["et"]))
    with torch.no_grad():
        old = original(x)
        initial = incremental(expanded)
    for task in original.tasks:
        torch.testing.assert_close(old[task], initial[task], atol=0, rtol=0)
    before = copy.deepcopy(incremental.state_dict())
    incremental.train()
    step(incremental, expanded, samples(["lai"]), optimizer(incremental))
    for name, value in incremental.state_dict().items():
        if incremental.parameter_owner(name) != "lai":
            assert torch.equal(value, before[name]), name
    assert any(not torch.equal(v, before[n]) for n, v in incremental.state_dict().items()
               if incremental.parameter_owner(n) == "lai")
    # Each stage-B task starts from the same knowledge version, then can be composed.
    merged = MODELS.build(make_config(tmp_path).model)
    load_adapter_bundle(merged, bundle)
    for task in original.tasks:
        branch = copy.deepcopy(original)
        branch.set_training_tasks([task], train_shared=False)
        step(branch, x, y, optimizer(branch))
        path = tmp_path / f"{task}.pth"
        save_adapter_bundle(branch, path, tasks=[task])
        load_adapter_bundle(merged, path, tasks=[task], load_shared=False)
        with torch.no_grad():
            torch.testing.assert_close(merged(x)[task], branch(x)[task], atol=0, rtol=0)
    # Checks happen before any tensor is copied.
    invalid = torch.load(bundle, weights_only=True)
    key = next(k for k in invalid["state_dict"] if ".shared." in k)
    invalid["state_dict"][key].add_(1)
    corrupt = tmp_path / "corrupt.pth"
    torch.save(invalid, corrupt)
    before = copy.deepcopy(merged.state_dict())
    with pytest.raises(ValueError, match="checksum"):
        load_adapter_bundle(merged, corrupt)
    assert all(torch.equal(v, before[n]) for n, v in merged.state_dict().items())
    step(original, x, y, optimizer(original))
    different = tmp_path / "different.pth"
    save_adapter_bundle(original, different)
    with pytest.raises(ValueError, match="different shared knowledge"):
        load_adapter_bundle(merged, different, tasks=["water"], load_shared=False)


def test_config_selection_preserves_old_models():
    choices = dict(hydrostate_v1="HydroStateModel", hydrostate_staged="StagedHydroStateModel",
                   hydrostate_multilevel="MultiLevelHydroStateModel",
                   hydrostate_multilevel_temporal_l2="MultiLevelTemporalHydroStateModel")
    for name, model_type in choices.items():
        cfg = Config.fromfile(f"configs/experiments/legacy/{name}.py")
        assert cfg.model.type == model_type
        assert "shared_rank" not in cfg.model.encoder
        assert MODELS.get(model_type) is not None
    work_dirs = set()
    for path in Path("configs").glob("hydrostate_shared_private*.py"):
        cfg = Config.fromfile(str(path))
        configure_data_contract(cfg)
        assert cfg.model.type == "SharedPrivateHydroStateModel"
        assert cfg.model.encoder.type == "SharedPrivateOlmoEarthEncoder"
        assert cfg.target_time_queries and cfg.model.water_soil_coupling is None
        assert set(cfg.model.heads) == set(cfg.model.tasks)
        assert "adapter" not in cfg.model and "interaction" not in cfg.model
        assert cfg.optim_wrapper.constructor == "SharedPrivateOptimWrapperConstructor"
        assert cfg.work_dir not in work_dirs
        work_dirs.add(cfg.work_dir)


def test_real_mmengine_runner_resume_and_bundle_tools(tmp_path, monkeypatch):
    from mmengine.runner import Runner
    from torch.utils.data import DataLoader

    from tools.legacy import compose_shared_private, export_shared_private

    # Same compatibility setting as the project's existing MMEngine launchers.
    monkeypatch.setenv("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    monkeypatch.setattr("mmengine.runner.runner.get_device", lambda: "cpu")
    cfg = make_config(tmp_path)
    model = MODELS.build(cfg.model)
    batch = dict(inputs=inputs(model.tasks), data_samples=samples())
    # Runner owns preprocessing: supply WF on the input grid here.
    batch["data_samples"]["water"] = torch.full((2, 1, 12, 12), .3)
    batch["data_samples"]["water_valid"] = torch.ones(2, 1, 12, 12, dtype=torch.bool)

    def runner(instance, directory, epochs, checkpoint=None):
        return Runner(
            model=instance, work_dir=str(directory),
            train_dataloader=DataLoader([batch, batch], batch_size=1,
                                        collate_fn=lambda rows: copy.deepcopy(rows[0])),
            train_cfg=dict(by_epoch=True, max_epochs=epochs),
            optim_wrapper=dict(type="OptimWrapper", accumulative_counts=2,
                               constructor="SharedPrivateOptimWrapperConstructor",
                               optimizer=dict(type="AdamW", lr=.002, weight_decay=.1)),
            default_hooks=dict(checkpoint=dict(type="CheckpointHook", interval=1),
                               logger=dict(type="LoggerHook", interval=1)),
            default_scope="hydrostate", visualizer=dict(type="Visualizer", vis_backends=[]),
            env_cfg=dict(dist_cfg=dict(backend="gloo")),
            load_from=str(checkpoint) if checkpoint else None, resume=checkpoint is not None,
        )

    uninterrupted = copy.deepcopy(model)
    first = runner(model, tmp_path / "first", 1)
    first.train()
    checkpoint = tmp_path / "first/epoch_1.pth"
    restored = MODELS.build(cfg.model)
    resumed = runner(restored, tmp_path / "resumed", 2, checkpoint)
    resumed.train()
    runner(uninterrupted, tmp_path / "uninterrupted", 2).train()
    assert resumed.epoch == 2
    for name, value in restored.state_dict().items():
        torch.testing.assert_close(value, uninterrupted.state_dict()[name], msg=name)
    config_path = tmp_path / "expanded.py"
    cfg.dump(str(config_path))
    bundle = tmp_path / "water.pth"
    monkeypatch.setattr("sys.argv", ["export", str(config_path), str(checkpoint), str(bundle),
                                    "--tasks", "water"])
    export_shared_private.main()
    composed = tmp_path / "composed.pth"
    monkeypatch.setattr("sys.argv", ["compose", str(config_path), str(checkpoint), str(composed),
                                    "--task-bundles", str(bundle)])
    compose_shared_private.main()
    result = torch.load(composed, weights_only=True)
    assert result["meta"]["composed_tasks"] == ["water"]
    assert "optimizer" not in result
    model.load_state_dict(result["state_dict"], strict=True)


def test_transfer_rejects_base_stats_rank_and_support_drift(tmp_path):
    model = MODELS.build(make_config(tmp_path).model)
    bundle = tmp_path / "initial.pth"
    save_adapter_bundle(model, bundle)
    for field, value, message in (
        ("model.encoder.private_rank.water", 4, "Private rank"),
        ("model.target_statistics.tasks.water.std", 2., "statistics"),
        ("model.heads.water.activation", "identity", "Prediction head"),
    ):
        cfg = make_config(tmp_path)
        cfg.model.encoder.private_rank = dict.fromkeys(cfg.model.tasks, 2)
        cfg.merge_from_dict({field: value})
        changed = MODELS.build(cfg.model)
        with pytest.raises(ValueError, match=message):
            load_adapter_bundle(changed, bundle)
    # Fingerprints must not remain cached after an external checkpoint load.
    base_parameter = next(p for n, p in model.named_parameters()
                          if n.startswith("encoder.backbone.") and model.parameter_owner(n) is None)
    with torch.no_grad():
        base_parameter.add_(1)
    with pytest.raises(ValueError, match="foundation weights"):
        load_adapter_bundle(model, bundle)


@pytest.mark.parametrize("shared,private", [
    (0, 2), (2, 0), (2, dict(water=2, soil_moisture=4, et=8)),
])
def test_rank_controls_and_private_only_ablation(tmp_path, shared, private):
    cfg = make_config(tmp_path)
    cfg.model.encoder.update(shared_rank=shared, private_rank=private)
    model = MODELS.build(cfg.model)
    losses = model(inputs(model.tasks), samples(), mode="loss")
    losses["loss_water"].backward()
    for name, parameter in model.named_parameters():
        if model.parameter_owner(name) not in ("__shared__", "water"):
            assert parameter.grad is None, name
    layer = model.encoder.backbone.model.blocks[0].attn.q
    assert (layer.shared is not None) == bool(shared)
    assert len(layer.private) == (0 if private == 0 else 3)


def distributed_worker(rank, directory):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    torch.set_num_threads(1)
    torch.manual_seed(48)
    dist.init_process_group("gloo", init_method=f"file://{directory}/ddp", rank=rank, world_size=2)
    try:
        with pytest.MonkeyPatch.context() as monkeypatch:
            install_tiny_backbone(monkeypatch)
            model = MODELS.build(make_config(Path(directory)).model)
            wrapped = DistributedDataParallel(model, find_unused_parameters=True)
            optim = optimizer(model)
            x, y = inputs(model.tasks), samples()
            # Two consecutive globally empty batches must not leave DDP waiting.
            for iteration in range(4):
                labels = copy.deepcopy(y)
                if rank == 1 or iteration in (1, 2):
                    for name, value in labels.items():
                        if name.endswith("_valid"):
                            value.zero_()
                losses = step(wrapped, x, labels, optim)
                assert ("loss_empty" in losses) == (iteration in (1, 2))
            for parameter in model.parameters():
                other = parameter.detach().clone()
                dist.broadcast(other, src=0)
                torch.testing.assert_close(parameter, other)
    finally:
        dist.destroy_process_group()


def test_ddp_unequal_labels_and_empty_batches(tmp_path):
    import torch.multiprocessing as mp

    mp.spawn(distributed_worker, args=(str(tmp_path),), nprocs=2, join=True)


def test_regional_checkpoint_and_explicit_physical_alignment(tmp_path):
    from hydrostate.inference.predict import load_model, masked_outputs

    cfg = make_config(tmp_path)
    model = MODELS.build(cfg.model).eval()
    x = inputs(model.tasks)
    checkpoint = tmp_path / "model.pth"
    torch.save(dict(state_dict=model.state_dict()), checkpoint)
    statistics = tmp_path / "stats.json"
    statistics.write_text(json.dumps(cfg.model.target_statistics.to_dict()))
    config_path = tmp_path / "model.py"
    cfg.dump(str(config_path))
    plan = dict(config=dict(input_profile="landsat_l2_sr_v1", pretrained=str(tmp_path),
                            target_statistics=str(statistics)))
    restored, _ = load_model(plan, checkpoint=checkpoint, model_config=config_path, device="cpu")
    assert restored.requires_target_time
    with torch.no_grad():
        expected, actual = model(x), restored(x)
        masked = masked_outputs(restored, x)
    for task in model.tasks:
        torch.testing.assert_close(expected[task], actual[task], atol=0, rtol=0)
        assert masked[task].shape == (2, 4, 4)
    cfg.model.water_soil_coupling = dict(enabled=True, supervision_source="observed")
    coupled = MODELS.build(cfg.model)
    x["target_time"]["soil_moisture"] += 86400
    with pytest.raises(RuntimeError, match="identical WF and SM query times"):
        coupled(x)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is needed for BF16 AMP")
@pytest.mark.parametrize("gate_mode", ["learned", "fixed_one"])
def test_cuda_bfloat16_training_uses_only_adaptation_parameters(tmp_path, gate_mode):
    cfg = make_config(tmp_path)
    cfg.model.encoder.shared_gate_mode = gate_mode
    model = MODELS.build(cfg.model).cuda().train()
    cfg.optim_wrapper.accumulative_counts = 1
    optim = build_optim_wrapper(model, cfg.optim_wrapper)
    batch = dict(inputs=inputs(model.tasks), data_samples=samples())
    batch["data_samples"]["water"] = torch.full((2, 1, 12, 12), .3)
    batch["data_samples"]["water_valid"] = torch.ones(2, 1, 12, 12, dtype=torch.bool)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    for _ in range(2):
        result = model.train_step(copy.deepcopy(batch), optim)
        assert torch.isfinite(result["loss"])
    for name, parameter in model.named_parameters():
        assert torch.isfinite(parameter).all(), name
        if model.parameter_owner(name) is None:
            assert torch.equal(before[name], parameter), name
    assert any(not torch.equal(before[n], p) for n, p in model.named_parameters()
               if model.parameter_owner(n) == "__shared__")


def test_fixed_one_has_no_gate_and_preserves_legacy_bundle_contract(tmp_path):
    layer = SharedPrivateLinear(torch.nn.Linear(4, 3), 0, 2, dict(water=0))
    torch.nn.init.normal_(layer.shared.b.weight)
    x = torch.randn(2, 4)
    with task_context(TaskContext("water", None)):
        torch.testing.assert_close(layer(x), layer.base(x) + layer.shared(x))
    cfg = make_config(tmp_path)
    old = MODELS.build(cfg.model)
    source = tmp_path / "legacy.pth"
    save_adapter_bundle(old, source)
    bundle = torch.load(source, weights_only=True)
    del bundle["encoder"]["shared_gate_mode"]
    torch.save(bundle, source)
    load_adapter_bundle(old, source)
    cfg.model.encoder.shared_gate_mode = "fixed_one"
    fixed = MODELS.build(cfg.model)
    assert not fixed.encoder.gates
    with pytest.raises(ValueError, match="shared_gate_mode"):
        load_adapter_bundle(fixed, source)
    losses = fixed(inputs(fixed.tasks), samples(), mode="loss")
    sum(v for k, v in losses.items() if k.startswith("loss_")).backward()
    assert fixed.encoder.backbone.model.blocks[0].attn.q.shared.b.weight.grad.abs().sum() > 0
    save_adapter_bundle(fixed, tmp_path / "fixed.pth")
    restored = MODELS.build(cfg.model)
    load_adapter_bundle(restored, tmp_path / "fixed.pth")


def test_three_modes_have_identical_initial_readouts(tmp_path):
    branches = []
    for shared, private in ((4, 0), (2, 2), (0, 4)):
        cfg = make_config(tmp_path)
        cfg.model.encoder.update(shared_rank=shared, private_rank=private,
                                 shared_gate_mode="fixed_one")
        cfg.model.branch_init_seed = 3407
        model = MODELS.build(cfg.model)
        branches.append({n: v for n, v in model.state_dict().items()
                         if n.startswith(("queries.", "decoders.", "heads.",
                                          "encoder.landsat_l2_adapter."))})
        assert not model.encoder.gates
    for other in branches[1:]:
        assert other.keys() == branches[0].keys()
        for name, value in other.items():
            torch.testing.assert_close(value, branches[0][name], atol=0, rtol=0)


def test_task_only_execution_cannot_receive_other_task_gradients(tmp_path):
    cfg = make_config(tmp_path)
    cfg.model.encoder.update(shared_rank=0, private_rank=4, shared_gate_mode="fixed_one")
    cfg.model.update(train_tasks=["soil_moisture"], train_shared=False, active_tasks_only=True)
    single = MODELS.build(cfg.model).train()
    reference = copy.deepcopy(single)
    reference.active_tasks_only = False
    x, y = inputs(single.tasks), samples()
    visited = []
    handles = [single.decoders[t].register_forward_hook(
        lambda module, args, output, t=t: visited.append(t)) for t in single.tasks]
    optim = build_optim_wrapper(single, dict(
        type="OptimWrapper", constructor="SharedPrivateOptimWrapperConstructor",
        clip_grad=dict(max_norm=1.0), optimizer=dict(type="AdamW", lr=.002, weight_decay=.1)))
    other_optim = build_optim_wrapper(reference, dict(
        type="OptimWrapper", constructor="SharedPrivateOptimWrapperConstructor",
        clip_grad=dict(max_norm=1.0), optimizer=dict(type="AdamW", lr=.002, weight_decay=.1)))
    losses = step(single, x, y, optim)
    assert visited == ["soil_moisture"]
    assert {k for k in losses if k.startswith("loss_")} == {"loss_soil_moisture"}
    step(reference, x, y, other_optim)
    for name, value in single.state_dict().items():
        torch.testing.assert_close(value, reference.state_dict()[name], atol=0, rtol=0)
    assert all(single.parameter_owner(n) == "soil_moisture"
               for n, p in single.named_parameters() if p.requires_grad)
    prediction = single.predict(x, y)[0]
    assert "soil_moisture" in prediction and "water" not in prediction and "et" not in prediction
    for handle in handles:
        handle.remove()
    empty = copy.deepcopy(y)
    empty["soil_moisture_valid"].zero_()
    assert "loss_empty" in step(single, x, empty, optim)


def test_single_task_metric_exports_reproducible_sample_errors(tmp_path):
    from hydrostate.evaluation import HydroStateMetric

    cfg = make_config(tmp_path)
    cfg.model.encoder.update(shared_rank=0, private_rank=4, shared_gate_mode="fixed_one")
    cfg.model.update(train_tasks=["soil_moisture"], train_shared=False, active_tasks_only=True)
    model = MODELS.build(cfg.model)
    y = samples()
    y["metainfo"] = dict(sample_id=["first", "second"])
    predictions = model.predict(inputs(model.tasks), y)
    path = tmp_path / "samples.jsonl"
    metric = HydroStateMetric(cfg.model.target_statistics, tasks=cfg.model.tasks,
                              evaluated_tasks=["soil_moisture"], sample_metrics_path=str(path))
    metric.process(None, predictions)
    scores = metric.compute_metrics(metric.results)
    assert "soil_moisture_rmse" in scores and "water_rmse" not in scores
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["sample_id"] for r in rows] == ["first", "second"]
    expected = (sum(r["soil_moisture/sum_sq"] for r in rows) /
                sum(r["soil_moisture/count"] for r in rows)) ** .5
    assert scores["soil_moisture_rmse"] == expected


@pytest.mark.parametrize("single", [True, False])
def test_ablation_evaluation_runner_writes_final_and_best_results(tmp_path, monkeypatch, single):
    from mmengine.runner import Runner
    from torch.utils.data import DataLoader

    from tools.legacy.evaluate_sharing_ablation import evaluate

    monkeypatch.setenv("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    monkeypatch.setattr("mmengine.runner.runner.get_device", lambda: "cpu")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    cfg = make_config(tmp_path)
    cfg.model.encoder.update(shared_rank=0 if single else 2, private_rank=4 if single else 2,
                             shared_gate_mode="fixed_one")
    cfg.model.update(train_tasks=["soil_moisture"] if single else None,
                     train_shared=not single, active_tasks_only=single)
    cfg.train_cfg.max_epochs = 50
    metric = "soil_moisture_rmse" if single else "hydro_score"
    cfg.default_hooks.checkpoint.save_best = metric
    cfg.visualizer = dict(type="Visualizer", vis_backends=[])
    cfg.custom_hooks = []
    path = tmp_path / "independent.py"
    cfg.dump(str(path))
    model = MODELS.build(cfg.model)
    for name in ("epoch_50.pth", f"best_{metric}_epoch_50.pth"):
        torch.save(dict(state_dict=model.state_dict(), meta=dict(epoch=50)), tmp_path / name)
    y = samples()
    y["water"] = torch.full((2, 1, 12, 12), .3)
    y["water_valid"] = torch.ones(2, 1, 12, 12, dtype=torch.bool)
    y["metainfo"] = dict(sample_id=["first", "second"])
    batch = dict(inputs=inputs(model.tasks), data_samples=y)
    def from_cfg(config):
        return Runner(
            model=MODELS.build(config.model), work_dir=config.work_dir,
            test_dataloader=DataLoader([0, 1], batch_size=2,
                                       collate_fn=lambda rows: copy.deepcopy(batch)),
            test_cfg=dict(type="TestLoop"), test_evaluator=config.test_evaluator,
            load_from=config.load_from, default_scope="hydrostate",
            visualizer=dict(type="Visualizer", vis_backends=[]), cfg=config)

    monkeypatch.setattr(Runner, "from_cfg", from_cfg)
    report = evaluate(path, tmp_path)
    saved = json.loads((tmp_path / "evaluation/metrics.json").read_text())
    assert saved["selected_tasks"] == (["soil_moisture"] if single else
                                       ["water", "soil_moisture", "et"])
    assert set(saved["results"]) == {"final", "best"}
    for kind in saved["results"]:
        result = saved["results"][kind]
        assert ("water_rmse" not in result["metrics"]) == single
        assert result["metrics"] == report["results"][kind]["metrics"]
        assert len(Path(result["sample_metrics"]).read_text().splitlines()) == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is needed for empty BF16 windows")
def test_independent_bf16_empty_accumulation_preserves_adam_state(tmp_path):
    cfg = make_config(tmp_path)
    cfg.model.encoder.update(shared_rank=0, private_rank=4, shared_gate_mode="fixed_one")
    cfg.model.update(train_tasks=["soil_moisture"], train_shared=False, active_tasks_only=True)
    cfg.optim_wrapper.update(accumulative_counts=2, loss_scale=dict(enabled=False))
    model = MODELS.build(cfg.model).cuda().train()
    optim = build_optim_wrapper(model, cfg.optim_wrapper)
    y = samples()
    y["water"] = torch.full((2, 1, 12, 12), .3)
    y["water_valid"] = torch.ones(2, 1, 12, 12, dtype=torch.bool)
    batch = dict(inputs=inputs(model.tasks), data_samples=y)
    empty = copy.deepcopy(batch)
    empty["data_samples"]["soil_moisture_valid"].zero_()
    for current in (empty, batch, empty, batch):
        before = copy.deepcopy(model.state_dict())
        state = copy.deepcopy(optim.optimizer.state_dict()["state"])
        for _ in range(2):
            losses = model.train_step(copy.deepcopy(current), optim)
            assert torch.isfinite(losses["loss"])
        if current is empty:
            assert all(torch.equal(v, before[n]) for n, v in model.state_dict().items())
            after = optim.optimizer.state_dict()["state"]
            assert state.keys() == after.keys()
            for key in state:
                for field in state[key]:
                    torch.testing.assert_close(state[key][field], after[key][field], atol=0, rtol=0)
        else:
            assert any(not torch.equal(v, before[n]) for n, v in model.state_dict().items())
