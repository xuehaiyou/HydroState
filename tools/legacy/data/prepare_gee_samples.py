"""Prepare small samples with direct pixels (legacy Drive remains optional)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from hydrostate.legacy.sampling import main

if __name__ == "__main__":
    main()
