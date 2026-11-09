#!/usr/bin/env python
"""Check a configured retrieval model on a bounded real training subset."""

import argparse

from hydrostate.experiments.retrieval_smoke import smoke


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    smoke(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
