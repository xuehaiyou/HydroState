"""Prepare, convert, partition and publish manual coverage proportions."""

import argparse
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "labels", "collect", "audit-training", "audit"):
        p = sub.add_parser(name)
        p.add_argument("--root", required=True)
        if name == "prepare":
            p.add_argument("--glacier-root", required=True)
            p.add_argument("--flood-root", required=True)
        if name == "collect":
            p.add_argument("--project", required=True)
            p.add_argument("--concurrency", type=int, default=4)
        if name in {"collect", "audit"}:
            p.add_argument("--pilot", action="store_true")
        if name == "audit":
            p.add_argument("--publish", action="store_true")
        if name == "audit-training":
            p.add_argument("--manifests", action="append", required=True)
    args = vars(parser.parse_args())
    name = args.pop("command")
    from .pipeline import collect, labels
    from .prepare import prepare
    from .validation import audit, audit_training

    function = {
        "prepare": prepare,
        "labels": labels,
        "collect": collect,
        "audit": audit,
        "audit-training": audit_training,
    }[name]
    print(json.dumps(function(**args), allow_nan=False), flush=True)
