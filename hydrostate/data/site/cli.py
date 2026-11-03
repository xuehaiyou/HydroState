"""Explicit, resumable station data preparation stages."""

import argparse
import json
from pathlib import Path

from hydrostate.data.gee.contracts import atomic_json

from .archives import extract, inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    for name in ("inventory", "extract"):
        command = stages.add_parser(name)
        command.add_argument("--root", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--pattern", default="*/*.zip")
    args = parser.parse_args()
    paths = sorted(args.root.glob(args.pattern))
    if not paths:
        parser.error("No source archives matched")
    if args.stage == "inventory":
        records = inventory(paths)
    else:
        records = []
        for index, path in enumerate(paths, 1):
            record = extract(path)
            records.append(record)
            atomic_json(args.output, dict(complete=False, archives=records))
            print(json.dumps(dict(index=index, total=len(paths), **record)), flush=True)
    atomic_json(args.output, dict(complete=True, archives=records))


if __name__ == "__main__":
    main()
