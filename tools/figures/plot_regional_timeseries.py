"""Summarize a regional product or plot its exported time-series CSV."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from hydrostate.inference.regional_timeseries import plot, summarize  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('command', choices=['summarize', 'plot'])
parser.add_argument('--input', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
(summarize if args.command == 'summarize' else plot)(args.input, args.output)
