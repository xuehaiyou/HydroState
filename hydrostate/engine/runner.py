"""MMEngine runner with explicit loading of trusted local training checkpoints."""

from pathlib import Path

import torch
from mmengine.runner import Runner
from mmengine.runner.checkpoint import CheckpointLoader

_LOCAL_PREFIX = "hydrostate-local://"


@CheckpointLoader.register_scheme(prefixes=_LOCAL_PREFIX)
def load_local_training_checkpoint(filename, map_location):
    """Restore project checkpoints, including MMEngine history and optimizer metadata.

    These are trusted local training artifacts. Remote/model-zoo loaders and ordinary
    torch.load calls retain their own policies; no global torch setting is changed.
    """
    path = Path(filename.removeprefix(_LOCAL_PREFIX)).expanduser()
    if not path.is_file():
        raise FileNotFoundError(path)
    return torch.load(path, map_location=map_location, weights_only=False)


class HydroStateRunner(Runner):
    """Use full-state deserialization for explicitly selected local checkpoints.

    Both explicit loading and Runner.resume (including automatic resume) pass here.
    Only use local checkpoints produced by this project or another trusted source.
    """

    def load_checkpoint(self, filename, *args, **kwargs):
        filename = str(filename)
        if "://" not in filename:
            filename = _LOCAL_PREFIX + str(Path(filename).expanduser().resolve())
        return super().load_checkpoint(filename, *args, **kwargs)
