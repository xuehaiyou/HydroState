"""Bind current model configurations to one declared data contract."""

from hydrostate.tasks import task_config

from .contract import OUTPUT_SCALE, PROFILE, TASKS, VERSION


def require_current_config(cfg):
    """Reject historical data settings before loading arrays or model weights."""
    if cfg.get("data_contract") != VERSION:
        raise ValueError(f"Expected data_contract={VERSION}; use a current retrieval config")
    for section in (cfg, cfg.model):
        if task_config(section.get("tasks", TASKS)) != task_config(TASKS):
            raise ValueError("Current retrieval tasks disagree with the canonical contract")
        section.tasks = task_config(TASKS)
    if cfg.get("input_profile", PROFILE) != PROFILE:
        raise ValueError("Current retrieval input_profile disagrees with the contract")
    cfg.input_profile = PROFILE
    if cfg.model.get("output_scale", OUTPUT_SCALE) != OUTPUT_SCALE:
        raise ValueError("Current retrieval requires 30 m outputs on the declared grid")
    for key in ("train_dataloader", "val_dataloader", "test_dataloader"):
        if key in cfg and cfg[key].dataset.type != "RetrievalDataset":
            raise ValueError(f"{key} must use RetrievalDataset for {VERSION}")
