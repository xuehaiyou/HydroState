"""Profile native input reads, CUDA batch sizes, operators, and the regional pipeline."""

import argparse
import logging
from pathlib import Path

from hydrostate.inference.regional_benchmark import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "model-config", "checkpoint", "pretrained", "statistics", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--dates", nargs="+", required=True)
    parser.add_argument("--start-index", type=int, default=27000)
    parser.add_argument("--windows", type=int, default=32)
    parser.add_argument("--batches", nargs="+", type=int, default=[2, 4, 8, 16, 20, 24])
    parser.add_argument("--workers", nargs="+", type=int, default=[0, 2, 4])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.windows < 16 or args.repeats < 1 or min(args.batches) < 1 or min(args.workers) < 0:
        parser.error("Require >=16 windows, positive batches/repeats and nonnegative workers")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(args)


if __name__ == "__main__":
    main()
