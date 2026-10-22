from types import SimpleNamespace

import pytest
import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope

from hydrostate.cli import configure_data_contract
from hydrostate.data.retrieval.contract import TASKS
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


def batch():
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
        inputs[sensor] = torch.rand(2, 5, channels, 12, 12)
        inputs[f"{sensor}_valid"] = torch.ones(2, 5, 1, 12, 12, dtype=torch.bool)
        inputs[f"{sensor}_quality"] = torch.ones(2, 5, 4, 12, 12)
    labels = {}
    for task in ("water", "snow_ice", "land"):
        labels[task] = torch.full((2, 1, 4, 4), 1.0 if task == "land" else 0.0)
        labels[f"{task}_valid"] = torch.ones(2, 1, 4, 4, dtype=torch.bool)
    labels["soil_moisture"] = torch.full((2, 1, 1, 1), 0.2)
    labels["soil_moisture_valid"] = torch.ones(2, 1, 1, 1, dtype=torch.bool)
    labels["soil_moisture_domain_valid"] = torch.ones(2, 1, 4, 4, dtype=torch.bool)
    labels["soil_moisture_weight"] = torch.ones(2, 1, 1, 1)
    for task in ("et_water", "et_ice", "et_land"):
        labels[f"{task}_native_value"] = torch.ones(2, 2)
        labels[f"{task}_native_valid"] = torch.ones(2, 2, dtype=torch.bool)
        labels[f"{task}_native_weights"] = torch.full((2, 2, 4, 4), 1 / 16)
    return dict(inputs=inputs, data_samples=labels)


@pytest.mark.parametrize("private_rank", [0, 2])
@pytest.mark.parametrize("checkpoint", [False, True])
def test_current_model_forward_backward_and_prediction(
    monkeypatch, tmp_path, private_rank, checkpoint
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
        configure_data_contract(cfg)
        cfg.model.encoder.update(
            model_path=str(tmp_path),
            embedding_channels=32,
            out_indices=[1, 3, 5],
            shared_rank=2,
            private_rank=private_rank,
            gradient_checkpointing=checkpoint,
        )
        cfg.model.decoder.update(in_channels=32, channels=8)
        cfg.model.input_fusion.feature_channels = 32
        for head in cfg.model.heads.values():
            head.in_channels = 8
        cfg.model.target_statistics = statistics()
        model = MODELS.build(cfg.model).train()
        data = model.data_preprocessor(batch(), training=True)
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
