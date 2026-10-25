"""Prepare current regional five-slot inputs through computePixels or Google Drive."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hydrostate.data.regional.cli import prepare_main  # noqa: E402

if __name__ == "__main__":
    prepare_main()
