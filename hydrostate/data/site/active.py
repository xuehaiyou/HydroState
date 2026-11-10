"""Atomic active-release pointer for replacing ET without copying immutable SM data."""

import json
from pathlib import Path


def active_release(root):
    root = Path(root)
    pointer = root / "ACTIVE.json"
    if not pointer.exists():
        return root
    value = json.loads(pointer.read_text())
    if value.get("version") != "site_active_release_v1":
        raise ValueError("Unknown station active release pointer")
    target = (root / value["release"]).resolve()
    if not target.is_relative_to(root.resolve()) or target == root.resolve():
        raise ValueError("Active station release must be a child of the publication root")
    if not (target / "_PUBLISHED.json").exists():
        raise ValueError("Active station release has not completed publication")
    return target
