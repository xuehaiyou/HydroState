#!/usr/bin/env python
"""Publish already aligned NPZ observations as an immutable input or label pack."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from hydrostate.storage import VariablePackWriter


def build_pack(manifest, records, spec, output, shard_size=256):
    """Records have sample_id and npz_path; NPZ paths are relative to records."""
    records = Path(records)
    table = pd.read_parquet(records)
    if not {"sample_id", "npz_path"}.issubset(table):
        raise ValueError("Input records require sample_id and npz_path")
    if table.sample_id.duplicated().any():
        raise ValueError("Input records contain duplicate sample IDs")
    with VariablePackWriter(output, manifest, shard_size=shard_size, **spec) as writer:
        for row in table.itertuples():
            path = Path(row.npz_path)
            if not path.is_absolute():
                path = records.parent / path
            with np.load(path, allow_pickle=False) as values:
                writer.write(row.sample_id, values)
        return writer.finalize()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Existing samples.parquet")
    parser.add_argument("--records", required=True, help="Parquet with sample_id and npz_path")
    parser.add_argument("--spec", required=True, type=Path, help="JSON pack specification")
    parser.add_argument("--output", required=True, type=Path, help="New immutable pack directory")
    parser.add_argument("--shard-size", type=int, default=256)
    args = parser.parse_args()
    print(build_pack(args.manifest, args.records, json.loads(args.spec.read_text()),
                     args.output, args.shard_size))


if __name__ == "__main__":
    main()
