import torch

from hydrostate.models.site_supervision import station_losses, station_values


def test_point_sm_unchanged_by_land_fraction_and_independent_et_support():
    outputs = {
        name: torch.full((1, 1, 2, 2), value, requires_grad=True)
        for name, value in (
            ("soil_moisture", 0.3),
            ("water", 0.8),
            ("snow_ice", 0.1),
            ("land", 0.1),
            ("et_water", 5.0),
            ("et_ice", 2.0),
            ("et_land", 1.0),
        )
    }
    samples = dict(prediction_valid=torch.ones(1, 1, 2, 2, dtype=torch.bool))
    for name in ("site_sm", "site_et"):
        samples[f"{name}_native_value"] = torch.tensor([[0.2]])
        samples[f"{name}_native_valid"] = torch.tensor([[True]])
        samples[f"{name}_native_weights"] = torch.zeros(1, 1, 2, 2)
    samples["site_sm_native_weights"][0, 0, 0, 0] = 1
    samples["site_et_native_weights"][:] = 0.25
    projected = station_values(outputs, samples)
    torch.testing.assert_close(projected["site_sm"][0], torch.tensor([[0.3]]))
    torch.testing.assert_close(projected["site_et"][0], torch.tensor([[4.3]]))
    losses = station_losses(
        outputs, samples, {name: dict(std=1) for name in ("site_sm", "site_et")}, {}
    )
    losses["loss_site_sm"].backward(retain_graph=True)
    assert outputs["land"].grad is None
    assert outputs["soil_moisture"].grad[0, 0, 0, 0] > 0
    assert outputs["soil_moisture"].grad[0, 0, 1, 1] == 0
    losses["loss_site_et"].backward()
    assert outputs["water"].grad.abs().sum() > 0
    assert outputs["et_land"].grad.abs().sum() > 0
    samples["prediction_valid"][0, 0, 1, 1] = False
    # Missing an ET footprint pixel does not invalidate the separately supported SM point.
    values = station_values(outputs, samples)
    assert values["site_sm"][2].all() and not values["site_et"][2].any()
    samples["site_et_native_value"][:] = float("nan")
    samples["site_et_native_valid"][:] = False
    losses = station_losses(
        outputs, samples, {name: dict(std=1) for name in ("site_sm", "site_et")}, {}
    )
    assert losses["loss_site_et"] == 0
    assert torch.isfinite(losses["loss_site_sm"])
