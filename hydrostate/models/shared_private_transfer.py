"""Version-checked task transfer, distinct from an exact MMEngine resume."""

from __future__ import annotations

import copy
import hashlib
from dataclasses import asdict
from pathlib import Path

import torch

from hydrostate.tasks import task_config


def tensor_fingerprint(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        tensor = tensor.detach().cpu().contiguous()
        digest.update(f"{name}|{tensor.dtype}|{tuple(tensor.shape)}\n".encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _base_fingerprint(model):
    state = {name: value for name, value in model.state_dict().items()
             if name.startswith("encoder.backbone.model.")
             and model.parameter_owner(name) is None}
    return tensor_fingerprint(state)


def _plain(value):
    """Keep config metadata loadable with torch's weights_only safety default."""
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return copy.deepcopy(value)


def adapter_bundle(model, tasks=None):
    selected = tuple(model.tasks if tasks is None else tasks)
    if not selected or len(set(selected)) != len(selected) or not set(selected) <= set(model.tasks):
        raise ValueError("Export tasks must be unique configured task names")
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()
             if model.parameter_owner(name) in (*selected, "__shared__")}
    shared = {name: value for name, value in state.items()
              if model.parameter_owner(name) == "__shared__"}
    settings = model.encoder.adaptation_settings()
    settings["private_rank"] = {task: settings["private_rank"][task] for task in selected}
    stats = model.loss_module.target_statistics
    return dict(
        format="hydrostate.shared_private", version=1,
        base_fingerprint=_base_fingerprint(model), shared_fingerprint=tensor_fingerprint(shared),
        encoder=settings, output_scale=model.output_scale,
        water_soil_coupling=(asdict(model.water_soil_coupling)
                             if model.water_soil_coupling is not None else None),
        tasks={task: task_config(model.task_specs)[task] for task in selected},
        statistics={task: _plain(stats["tasks"][task]) for task in selected},
        branches=dict(decoder=_plain(model.branch_configs["decoder"]),
                      query_time=_plain(model.branch_configs["query_time"]),
                      heads={task: _plain(model.branch_configs["heads"][task])
                             for task in selected}),
        state_dict=state,
    )


def save_adapter_bundle(model, path, tasks=None):
    bundle = adapter_bundle(model, tasks)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Refuse overwriting a published knowledge version.
    with destination.open("xb") as stream:
        torch.save(bundle, stream)


def load_adapter_bundle(model, path, tasks=None, load_shared=True):
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    if bundle.get("format") != "hydrostate.shared_private" or bundle.get("version") != 1:
        raise ValueError("Unsupported shared/private adapter bundle")
    selected = tuple(bundle["tasks"] if tasks is None else tasks)
    if (len(set(selected)) != len(selected) or not set(selected) <= set(bundle["tasks"])
            or not set(selected) <= set(model.tasks)):
        raise ValueError("Import tasks must exist in both the bundle and destination model")
    current = model.encoder.adaptation_settings()
    # Version-1 bundles written before the fixed-coefficient ablation used gates.
    bundle["encoder"].setdefault("shared_gate_mode", "learned")
    for key in current:
        if key != "private_rank" and current[key] != bundle["encoder"][key]:
            raise ValueError(f"Incompatible encoder setting: {key}")
    if model.output_scale != bundle["output_scale"]:
        raise ValueError("Adapter bundle output grid differs")
    coupling = asdict(model.water_soil_coupling) if model.water_soil_coupling is not None else None
    if coupling != bundle["water_soil_coupling"]:
        raise ValueError("Adapter bundle WF-SM support rule differs")
    if _base_fingerprint(model) != bundle["base_fingerprint"]:
        raise ValueError("Adapter bundle requires the exact original foundation weights")
    for task in selected:
        if task_config(model.task_specs)[task] != bundle["tasks"][task]:
            raise ValueError(f"Observation support differs for {task}")
        if current["private_rank"][task] != bundle["encoder"]["private_rank"][task]:
            raise ValueError(f"Private rank differs for {task}; use a separately trained candidate")
        if model.loss_module.target_statistics["tasks"][task] != bundle["statistics"][task]:
            raise ValueError(f"Frozen task statistics differ for {task}")
        for key in ("decoder", "query_time"):
            if _plain(model.branch_configs[key]) != bundle["branches"][key]:
                raise ValueError(f"Task branch configuration differs: {key}")
        if _plain(model.branch_configs["heads"][task]) != bundle["branches"]["heads"][task]:
            raise ValueError(f"Prediction head differs for {task}")
    source = bundle["state_dict"]
    source_shared = {name: value for name, value in source.items()
                     if model.parameter_owner(name) == "__shared__"}
    if tensor_fingerprint(source_shared) != bundle["shared_fingerprint"]:
        raise ValueError("Shared knowledge checksum mismatch")
    destination = model.state_dict()
    if not load_shared:
        existing = {name: value for name, value in destination.items()
                    if model.parameter_owner(name) == "__shared__"}
        if tensor_fingerprint(existing) != bundle["shared_fingerprint"]:
            raise ValueError("Task branches refer to a different shared knowledge version")
    expected = {name for name in destination
                if model.parameter_owner(name) in selected
                or (load_shared and model.parameter_owner(name) == "__shared__")}
    chosen = {name: value for name, value in source.items() if name in expected}
    if set(chosen) != expected:
        raise ValueError(f"Incomplete adapter bundle: {sorted(expected - set(chosen))[:5]}")
    for name, value in chosen.items():
        if value.shape != destination[name].shape or value.dtype != destination[name].dtype:
            raise ValueError(f"Incompatible adapter tensor: {name}")
        if not torch.isfinite(value).all():
            raise ValueError(f"Non-finite adapter tensor: {name}")
    # Validate the entire import before changing any destination tensor. Target
    # normalization buffers stay untouched; old task moments were checked above.
    with torch.no_grad():
        for name, value in chosen.items():
            destination[name].copy_(value)
