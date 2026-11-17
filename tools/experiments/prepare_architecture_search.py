"""Write fixed nested few-sample selection plans, without changing source data."""

import argparse

from hydrostate.experiments.search_subset import prepare

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    prepare(args.config, args.output)
