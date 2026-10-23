"""Bind current model configurations to one declared data contract."""

from hydrostate.tasks import task_config

from .contract import OUTPUT_SCALE, PROFILE, TASKS, VERSION
from .specs import resolve_label_specs


def require_current_config(cfg):
    """Reject historical data settings before loading arrays or model weights."""
    if cfg.get("data_contract") != VERSION:
        raise ValueError(f"Expected data_contract={VERSION}; use a current retrieval config")
    definitions = resolve_label_specs(cfg.get("label_specs"))
    expected = task_config({name: spec.task() for name, spec in definitions.items()})
    for section in (cfg, cfg.model):
        actual = task_config(section.get("tasks", TASKS))
        if actual not in (task_config(TASKS), expected):
            raise ValueError("Current retrieval tasks disagree with the canonical contract")
        section.tasks = expected
    for name, spec in definitions.items():
        if name not in cfg.model.heads:
            if name in TASKS:
                raise ValueError(f"Missing built-in prediction head: {name}")
            cfg.model.heads[name] = dict(
                type="RegressionHead",
                in_channels=cfg.model.decoder.channels,
                activation=spec.activation,
            )
    if cfg.get("input_profile", PROFILE) != PROFILE:
        raise ValueError("Current retrieval input_profile disagrees with the contract")
    cfg.input_profile = PROFILE
    if cfg.model.get("output_scale", OUTPUT_SCALE) != OUTPUT_SCALE:
        raise ValueError("Current retrieval requires 30 m outputs on the declared grid")
    for key in ("train_dataloader", "val_dataloader", "test_dataloader"):
        if key in cfg and cfg[key].dataset.type != "RetrievalDataset":
            raise ValueError(f"{key} must use RetrievalDataset for {VERSION}")
        if key in cfg:
            dataset = cfg[key].dataset
            if task_config(dataset.get("tasks", TASKS)) not in (task_config(TASKS), expected):
                raise ValueError(f"{key} tasks disagree with label specifications")
            dataset.tasks = expected
            for option in ("label_specs", "label_packs"):
                value = cfg.get(option, {} if option == "label_specs" else [])
                if option in dataset and dataset[option] != value:
                    raise ValueError(f"{key} {option} disagree with shared configuration")
                dataset[option] = value
