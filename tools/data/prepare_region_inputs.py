"""Prepare regional inputs via configured Drive, direct pixels, or local rasters."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.predict_region import main

if __name__ == "__main__":
    sys.argv.insert(1, "prepare")
    selectors = {"--input-config", "--backend", "--download-config", "--local-config"}
    if not any(arg.split("=")[0] in selectors for arg in sys.argv):
        sys.argv.extend([
            "--input-config",
            str(Path(__file__).resolve().parents[2] / "configs/data/regional/gee_drive.json"),
        ])
    main()
