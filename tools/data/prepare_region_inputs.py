"""Prepare regional GEE inputs; defaults to regional Drive exports."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.predict_region import main

if __name__ == "__main__":
    sys.argv.insert(1, "prepare")
    if "--backend" not in sys.argv:
        sys.argv.extend(["--backend", "drive"])
    main()
