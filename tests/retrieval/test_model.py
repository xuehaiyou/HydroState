from types import SimpleNamespace

import pytest
import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope

from hydrostate.cli import configure_data_contract
from hydrostate.data.retrieval.contract import INPUT_GROUPS, TASKS
from hydrostate.normalization import CONFIGURED_SUPPORT_VERSION
from hydrostate.registry import MODELS
from hydrostate.tasks import task_config


def statistics():
    return dict(
        version=2,
        split="train",
        ddof=0,
        support_version=CONFIGURED_SUPPORT_VERSION,
        training_fingerprint="0" * 64,
        task_specs=task_config(TASKS),
        tasks={t: dict(count=100, mean=0.3, std=0.2) for t in TASKS},
    )


def batch(size=12):
    output_size = size // 3
    bounds = torch.tensor(
        [[1600000000 + i * 8 * 86400, 1600000000 + (i + 1) * 8 * 86400] for i in range(5)]
    )
    inputs = dict(
        sensor_valid=torch.ones(2, 5, 3, dtype=torch.bool),
        slot_bounds=bounds[None].repeat(2, 1, 1),
        observation_time=bounds.mean(-1, dtype=torch.float64).long()[None, :, None].repeat(2, 1, 3),
        target_time={t: bounds[2][None].repeat(2, 1) for t in TASKS},
    )
    for sensor, channels in (("s1", 2), ("s2", 12), ("landsat", 11)):
        inputs[sensor] = torch.rand(2, 5, channels, size, size)
        inputs[f"{sensor}_valid"] = torch.ones(2, 5, 1, size, size, dtype=torch.bool)
    for name in INPUT_GROUPS:
        inputs[f"{name}_quality"] = torch.ones(2, 5, 4, size, size)
        inputs[f"{name}_valid"] = torch.ones(2, 5, 1, size, size, dtype=torch.bool)
    labels = {}
    for task in ("water", "snow_ice", "land"):
        labels[task] = torch.full((2, 1, output_size, output_size), 1.0 if task == "land" else 0.0)
        labels[f"{task}_valid"] = torch.ones(2, 1, output_size, output_size, dtype=torch.bool)
    labels["soil_moisture"] = torch.full((2, 1, 1, 1), 0.2)
    labels["soil_moisture_valid"] = torch.ones(2, 1, 1, 1, dtype=torch.bool)
    labels["soil_moisture_domain_valid"] = torch.ones(
        2, 1, output_size, output_size, dtype=torch.bool)
    labels["soil_moisture_weight"] = torch.ones(2, 1, 1, 1)
    for task in ("et_water", "et_ice", "et_land"):
        labels[f"{task}_native_value"] = torch.ones(2, 2)
        labels[f"{task}_native_valid"] = torch.ones(2, 2, dtype=torch.bool)
        labels[f"{task}_native_weights"] = torch.full(
            (2, 2, output_size, output_size), 1 / output_size**2)
    return dict(inputs=inputs, data_samples=labels)


def test_quality_encoding_reuse_preserves_nonzero_outputs_and_gradients():
    from copy import deepcopy

    from hydrostate.models.composite_quality import CompositeQualityFusion

    fusion = CompositeQualityFusion(feature_channels=8, hidden_channels=4)
    torch.nn.init.normal_(fusion.layers[-1].weight, std=0.1)
    reference = deepcopy(fusion)
    inputs = batch()["inputs"]
    features = [torch.rand(2, 8, 3, 3), torch.rand(2, 8, 3, 3), torch.rand(2, 8, 2, 2)]
    cache = {}
    actual = [fusion(f, inputs, cache=cache) for f in features]
    expected = [reference(f, inputs) for f in features]
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    sum(a.square().sum() for a in actual).backward()
    sum(b.square().sum() for b in expected).backward()
    for a, b in zip(fusion.parameters(), reference.parameters()):
        torch.testing.assert_close(a.grad, b.grad)


@pytest.mark.parametrize(
    "private_rank,checkpoint,extended,finetuning",
    [(0, False, False, "lora"), (2, False, False, "lora"),
     (0, True, False, "lora"), (2, True, False, "lora"), (2, True, True, "lora"),
     (0, False, False, "full"), (0, True, False, "full")],
)
def test_current_model_forward_backward_and_prediction(
    monkeypatch, tmp_path, private_rank, checkpoint, extended, finetuning
):
    from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.nn.flexi_vit import EncoderConfig
    from rslearn.models.olmoearth_pretrain import model as wrapper

    def load(*args, **kwargs):
        return SimpleNamespace(
            encoder=EncoderConfig(
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
        )

    monkeypatch.setattr(wrapper, "load_model_from_path", load)
    init_default_scope("hydrostate")
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        cfg = Config.fromfile("configs/hydrostate_mtlora.py")
        if extended:
            cfg.label_specs = dict(
                swe=dict(
                    source="fixture",
                    band="SWE",
                    units="mm",
                    support="native",
                    resolution_m=500,
                    qa="fixture",
                    aggregation="target_period_mean",
                    activation="softplus",
                    valid_min=0,
                )
            )
        configure_data_contract(cfg)
        cfg.model.encoder.update(
            model_path=str(tmp_path),
            embedding_channels=32,
            out_indices=[1, 3, 5],
            shared_rank=0 if finetuning == "full" else 2,
            private_rank=private_rank,
            finetuning=finetuning,
            gradient_checkpointing=checkpoint,
        )
        cfg.model.decoder.update(in_channels=32, channels=8)
        cfg.model.input_fusion.feature_channels = 32
        for head in cfg.model.heads.values():
            head.in_channels = 8
        cfg.model.target_statistics = statistics()
        sample = batch()
        if extended:
            cfg.model.target_statistics["task_specs"] = task_config(cfg.model.tasks)
            cfg.model.target_statistics["tasks"]["swe"] = dict(count=10, mean=1, std=0.5)
            sample["inputs"]["target_time"]["swe"] = sample["inputs"]["target_time"][
                "et_land"
            ].clone()
            for suffix in ("value", "valid", "weights"):
                sample["data_samples"][f"swe_native_{suffix}"] = sample["data_samples"][
                    f"et_land_native_{suffix}"
                ].clone()
        model = MODELS.build(cfg.model).train()
        data = model.data_preprocessor(sample, training=True)
        output = model(data["inputs"])
        assert all(v.shape == (2, 1, 4, 4) for v in output.values())
        torch.testing.assert_close(
            output["water"] + output["snow_ice"] + output["land"], torch.ones(2, 1, 4, 4)
        )
        loss = model.loss(**data)
        assert all(torch.isfinite(v) for v in loss.values())
        sum(v for k, v in loss.items() if k.startswith("loss_")).backward()
        assert model.encoder.landsat_toa_adapter.weight.grad is not None
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        assert model.input_fusion.layers[-1].weight.grad.abs().sum() > 0
        if extended:
            assert model.heads["swe"].layers[-1].weight.grad.abs().sum() > 0
            assert loss["valid_count_swe"] > 0
        predictions = model.predict(**data)
        for p in predictions:
            torch.testing.assert_close(
                p["et_total"],
                p["water"] * p["et_water"] + p["snow_ice"] * p["et_ice"] + p["land"] * p["et_land"],
            )
            assert p["supervised_soil_moisture"].shape == (1, 1, 1)
        # No product labels are required for inference; unavailable pixels stay masked.
        for sensor in ("s1", "s2", "landsat"):
            data["inputs"][f"{sensor}_valid"][0, :, :, :3, :3] = False
        predicted = model.predict(data["inputs"])
        assert not predicted[0]["et_total_prediction_valid"][0, 0, 0]
        assert torch.isnan(predicted[0]["et_total"][0, 0, 0])
    finally:
        torch.set_num_threads(threads)


@pytest.fixture
def unet():
    init_default_scope("hydrostate")
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    cfg = Config.fromfile("configs/hydrostate_unet.py")
    configure_data_contract(cfg)
    cfg.model.encoder.channels = [8, 16, 32, 64]
    cfg.model.decoder.update(encoder_channels=[8, 16, 32, 64], channels=8)
    for head in cfg.model.heads.values():
        head.in_channels = 8
    cfg.model.target_statistics = statistics()
    try:
        yield MODELS.build(cfg.model), cfg
    finally:
        torch.set_num_threads(threads)


def test_unet_training_uses_shared_physical_support_and_all_heads(unet):
    model, _ = unet
    sample = batch(24)
    # One mixed output pixel: the SM target remains unchanged, weights use land area.
    sample["data_samples"]["water"][:, :, 0, 0] = 0.5
    sample["data_samples"]["land"][:, :, 0, 0] = 0.5
    data = model.data_preprocessor(sample, training=True)
    output = model(data["inputs"])
    assert set(output) == set(TASKS)
    assert all(value.shape == (2, 1, 8, 8) for value in output.values())
    torch.testing.assert_close(sum(output[t] for t in ("water", "snow_ice", "land")),
                               torch.ones(2, 1, 8, 8))
    assert all((output[t] >= 0).all() for t in ("et_water", "et_ice", "et_land"))
    losses = model.loss(**data)
    sum(value for key, value in losses.items() if key.startswith("loss_")).backward()
    assert all(p.requires_grad and p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters())
    assert all(model.heads[t].layers[-1].weight.grad.abs().sum() > 0 for t in TASKS)
    predictions = model.predict(**data)
    weights = sample["data_samples"]["land"]
    expected = (output["soil_moisture"] * weights).sum((-2, -1)) / weights.sum((-2, -1))
    for i, prediction in enumerate(predictions):
        torch.testing.assert_close(prediction["supervised_soil_moisture"].flatten(), expected[i])
        torch.testing.assert_close(prediction["et_total"], sum(
            output[task][i] * output[cover][i] for task, cover in
            (("et_water", "water"), ("et_ice", "snow_ice"), ("et_land", "land"))))
    # A missing task label only removes that task's supervision.
    data["data_samples"]["soil_moisture_valid"].zero_()
    missing = model.loss(**data)
    assert missing["valid_count_soil_moisture"] == 0
    torch.testing.assert_close(missing["loss_et_land"], losses["loss_et_land"])


def test_unet_ignores_missing_fill_values_and_masks_export(unet):
    from copy import deepcopy

    model, _ = unet
    sample = batch(24)
    for key in ("s1", "s2", "landsat", *INPUT_GROUPS):
        sample["inputs"][f"{key}_valid"][:, :, :, :3, :3] = False
    # Whole missing sensor, in addition to the missing output pixel.
    sample["inputs"]["s2_valid"].zero_()
    sample["inputs"]["sensor_valid"][:, :, 1] = False
    for name, group in INPUT_GROUPS.items():
        if group.sensor == "s2":
            sample["inputs"][f"{name}_valid"].zero_()
    a = model.data_preprocessor(sample)["inputs"]
    b = deepcopy(a)
    for sensor in ("s1", "s2", "landsat"):
        b[sensor].masked_fill_(~b[f"{sensor}_valid"], float("nan"))
    for name in INPUT_GROUPS:
        b[f"{name}_quality"].masked_fill_(~b[f"{name}_valid"], float("inf"))
    torch.testing.assert_close(model.encoder.pack_inputs(a), model.encoder.pack_inputs(b))
    for x, y in zip(model.predict(a), model.predict(b)):
        for task in TASKS:
            torch.testing.assert_close(x[task], y[task], equal_nan=True)
        assert not x["prediction_valid"][0, 0, 0]
        assert not x["soil_moisture_prediction_valid"][0, 0, 0]
        assert torch.isnan(x["et_total"][0, 0, 0])


def test_unet_preserves_order_and_rejects_noncentral_or_async_queries(unet):
    from copy import deepcopy

    model, _ = unet
    inputs = model.data_preprocessor(batch(24))["inputs"]
    original = model.encoder.pack_inputs(inputs)
    changed = deepcopy(inputs)
    changed["s1"] = changed["s1"].flip(1)
    assert not torch.equal(original, model.encoder.pack_inputs(changed))
    for task in ("soil_moisture", None):
        changed = deepcopy(inputs)
        for name in ([task] if task else TASKS):
            changed["target_time"][name] += 86400
        with pytest.raises(RuntimeError, match="aligned target|central target"):
            model(changed)
    changed = deepcopy(inputs)
    # Valid year-end-like shorter target, still anchored to the central slot.
    for name in TASKS:
        changed["target_time"][name][:, 1] -= 2 * 86400
    assert model(changed)["land"].shape == (2, 1, 8, 8)


def test_unet_initialization_needs_no_foundation_model_and_is_reproducible(unet, monkeypatch):
    import builtins

    _, cfg = unet
    original_import = builtins.__import__

    def reject_fm(name, *args, **kwargs):
        if name.startswith(("rslearn", "olmoearth")):
            raise AssertionError(f"U-Net attempted to import {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_fm)
    from hydrostate.cli import configure_pretrained_encoder

    configure_pretrained_encoder(cfg)
    torch.manual_seed(123)
    a = MODELS.build(cfg.model)
    torch.manual_seed(123)
    b = MODELS.build(cfg.model)
    for left, right in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    torch.manual_seed(124)
    c = MODELS.build(cfg.model)
    assert not torch.equal(next(a.parameters()), next(c.parameters()))


def test_unet_full_grid_and_regional_checkpoint_roundtrip(unet, tmp_path):
    import json

    from hydrostate.data.retrieval.contract import PROFILE
    from hydrostate.inference.regional import load_model, predict_records

    _, cfg = unet
    # Exercise the production widths and 120 -> 40 grid once on CPU.
    cfg.model.encoder.channels = [64, 128, 256, 512]
    cfg.model.decoder.update(encoder_channels=[64, 128, 256, 512], channels=64)
    for head in cfg.model.heads.values():
        head.in_channels = 64
    model = MODELS.build(cfg.model).eval()
    sample = batch(120)
    record = dict(inputs={k: ({t: v[0] for t, v in value.items()}
                             if isinstance(value, dict) else value[0])
                          for k, value in sample["inputs"].items()})
    expected = predict_records(model, [record], "cpu")[0]
    assert expected["water"].shape == (1, 40, 40)
    config, checkpoint, stats = (
        tmp_path / name for name in ("config.py", "model.pth", "stats.json"))
    cfg.dump(str(config))
    torch.save(dict(state_dict=model.state_dict()), checkpoint)
    stats.write_text(json.dumps(dict(statistics(), data_contract=dict(input_profile=PROFILE))))
    loaded, provenance = load_model(config, checkpoint, None, stats, "cpu")
    actual = predict_records(loaded, [record], "cpu")[0]
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], equal_nan=True)
    assert "model_path" not in provenance["model_config"]["encoder"]


def test_fm_preflight_still_requires_local_weights(tmp_path):
    from hydrostate.cli import configure_pretrained_encoder

    cfg = Config.fromfile("configs/hydrostate_mtlora.py")
    with pytest.raises(FileNotFoundError, match="config.json"):
        configure_pretrained_encoder(cfg, tmp_path)
    (tmp_path / "config.json").write_text("{}")
    with pytest.raises(FileNotFoundError, match="weights.pth"):
        configure_pretrained_encoder(cfg, tmp_path)
    (tmp_path / "weights.pth").write_bytes(b"fixture")
    configure_pretrained_encoder(cfg, tmp_path)
    assert cfg.model.encoder.model_path == str(tmp_path)


def test_unet_bfloat16_optimizer_step_and_batch_independence(unet):
    model, _ = unet
    data = model.data_preprocessor(batch(24), training=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    before = next(model.parameters()).detach().clone()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = sum(v for k, v in model.loss(**data).items() if k.startswith("loss_"))
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    assert not torch.equal(before, next(model.parameters()))
    assert all(torch.isfinite(p).all() for p in model.parameters())
    # GroupNorm must not couple a sample's predictions to its batch neighbours.
    model.eval()
    batched = model.predict(data["inputs"])[0]
    single = batch(24)
    for key, value in single["inputs"].items():
        if isinstance(value, dict):
            single["inputs"][key] = {t: data["inputs"][key][t][:1] for t in value}
        else:
            single["inputs"][key] = data["inputs"][key][:1]
    inputs = model.data_preprocessor(dict(inputs=single["inputs"], data_samples={}))["inputs"]
    alone = model.predict(inputs)[0]
    for task in TASKS:
        torch.testing.assert_close(batched[task], alone[task], rtol=1e-5, atol=1e-6, equal_nan=True)


def test_unet_season_phase_uses_utc_calendar_year(unet):
    import math
    from datetime import UTC, datetime

    model, _ = unet
    for date, phase in (((2020, 1, 1), 0.0), ((2020, 7, 2), 0.5),
                        ((2021, 7, 2), 182 / 365)):
        sample = batch(24)
        start = int(datetime(*date, tzinfo=UTC).timestamp())
        shift = start - int(sample["inputs"]["slot_bounds"][0, 2, 0])
        sample["inputs"]["slot_bounds"] += shift
        sample["inputs"]["observation_time"] += shift
        for task in TASKS:
            sample["inputs"]["target_time"][task] += shift
        data = model.data_preprocessor(sample)
        torch.testing.assert_close(data["inputs"]["target_season"], torch.tensor(
            [[math.sin(2 * math.pi * phase), math.cos(2 * math.pi * phase)]] * 2))
