"""Run a current retrieval checkpoint on prepared regional inputs."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hydrostate.data.regional.cli import predict_main  # noqa: E402

if __name__ == "__main__":
    predict_main()
