"""Prepare metadata-only ET split revision and same-station date pool."""

import argparse
import json

from hydrostate.data.site.et_rebalance import prepare_revision

if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    for name in ("published", "audit", "parent", "output", "split-source"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--additional", type=int, default=1200)
    print(json.dumps(prepare_revision(**vars(parser.parse_args())), indent=2))
