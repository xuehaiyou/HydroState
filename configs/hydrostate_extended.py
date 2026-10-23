"""Current retrieval plus prepared labels; the same JSON drives writing and training."""

import json
import pathlib

_base_ = ["./hydrostate_mtlora.py"]
_spec_file = "{{$HYDROSTATE_LABEL_SPEC:}}"
_pack_path = "{{$HYDROSTATE_LABEL_PACK:}}"
if not _spec_file or not _pack_path:
    raise ValueError("Set HYDROSTATE_LABEL_SPEC and HYDROSTATE_LABEL_PACK for extension training")
_extension = json.loads(pathlib.Path(_spec_file).read_text())
label_specs = _extension["label_specs"]
label_packs = [dict(path=_pack_path, variables=list(label_specs))]
