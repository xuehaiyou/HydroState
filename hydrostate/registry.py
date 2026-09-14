"""Project registries inherited from MMEngine root registries."""

from mmengine.registry import (
    DATASETS as MMENGINE_DATASETS,
    DATA_SAMPLERS as MMENGINE_DATA_SAMPLERS,
    HOOKS as MMENGINE_HOOKS,
    METRICS as MMENGINE_METRICS,
    MODELS as MMENGINE_MODELS,
    OPTIM_WRAPPER_CONSTRUCTORS as MMENGINE_OPTIM_WRAPPER_CONSTRUCTORS,
    Registry,
)

MODELS = Registry("model", parent=MMENGINE_MODELS, locations=["hydrostate.models"])
DATASETS = Registry("dataset", parent=MMENGINE_DATASETS, locations=["hydrostate.datasets"])
DATA_SAMPLERS = Registry(
    "data sampler", parent=MMENGINE_DATA_SAMPLERS, locations=["hydrostate.datasets"]
)
METRICS = Registry("metric", parent=MMENGINE_METRICS, locations=["hydrostate.evaluation"])
HOOKS = Registry("hook", parent=MMENGINE_HOOKS, locations=["hydrostate.engine"])
# MMEngine's build_optim_wrapper queries this root registry directly, so custom
# optimizer constructors must be registered on it rather than a child registry.
OPTIM_WRAPPER_CONSTRUCTORS = MMENGINE_OPTIM_WRAPPER_CONSTRUCTORS
