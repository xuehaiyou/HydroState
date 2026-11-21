"""Prepare, collect and publish human cover point samples."""

import argparse
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--source", required=True)
    p.add_argument("--root", required=True)
    p = sub.add_parser("collect")
    p.add_argument("--root", required=True)
    p.add_argument("--project", required=True)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--pilot", action="store_true")
    p = sub.add_parser("audit")
    p.add_argument("--root", required=True)
    p.add_argument("--pilot", action="store_true")
    p.add_argument("--publish", action="store_true")
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "prepare":
        from .prepare import prepare

        result = prepare(**args)
    elif command == "collect":
        from .collection import collect

        result = collect(**args)
    else:
        from .validation import audit

        result = audit(**args)
    print(json.dumps(result, allow_nan=False), flush=True)
