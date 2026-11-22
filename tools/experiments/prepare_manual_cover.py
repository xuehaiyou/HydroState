import argparse
import json

from hydrostate.experiments.manual_cover import prepare

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ("models", "manual-root", "output"):
        p.add_argument("--" + name, required=True)
    print(json.dumps(prepare(**vars(p.parse_args()))))
