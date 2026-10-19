"""Resume Landsat L2 acquisition matching and publish an ordinary Zarr input pack."""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from hydrostate.data.gee.contracts import locked
from hydrostate.data.gee.workflows.landsat_l2 import PRODUCER, fetch_sample
from hydrostate.data.input_products import L2_PROFILE, landsat_l2_spec
from hydrostate.storage import VariablePackWriter
from hydrostate.storage.packs import file_hash, validate_samples
from tools.data.extend_gee_labels import Checkpoint, download, select_rows


def publish(rows, cache, args, contract):
    report = dict(samples=len(rows), samples_with_l2=0, valid_slots=0,
                  missing_l2_slots=0, contract=contract)
    with VariablePackWriter(
        args.output, args.manifest, kind="input", variables={"landsat_l2": landsat_l2_spec()},
        source=contract, version=PRODUCER, shard_size=256, chunk_samples=1,
    ) as writer:
        for index, row in enumerate(rows):
            values, provenance = cache.read(row['sample_id'])
            writer.write(row['sample_id'], values, provenance=provenance)
            valid = values['landsat_l2_valid'].any(axis=(1, 2, 3))
            report['samples_with_l2'] += int(valid.any())
            report['valid_slots'] += int(valid.sum())
            report['missing_l2_slots'] += len(provenance['missing_l2_slots'])
            if (index + 1) % 256 == 0:
                logging.info("Wrote L2 Zarr %d/%d", index + 1, len(rows))
        if file_hash(args.manifest) != contract['manifest_sha256']:
            raise RuntimeError("Manifest changed during collection")
        (writer.partial / 'coverage.json').write_text(json.dumps(report, indent=2) + '\n')
        writer.finalize()
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--work-dir', type=Path)
    p.add_argument('--project', default='cropland-477906')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--attempts', type=int, default=3)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--seed', type=int, default=43)
    p.add_argument('--publish-only', action='store_true')
    args = p.parse_args()
    if not 1 <= args.workers <= 32 or args.attempts < 1 or args.limit < 0:
        p.error('Invalid worker/attempt/limit count')
    if args.output.exists():
        raise FileExistsError(f'Refusing to replace published pack: {args.output}')
    args.work_dir = args.work_dir or args.output.with_name('.' + args.output.name + '.work')
    args.work_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(args.work_dir / 'run.log')])
    table = pd.read_parquet(args.manifest)
    validate_samples(table)
    rows = select_rows(table, args.limit, args.seed)
    contract = dict(producer=PRODUCER, manifest_sha256=file_hash(args.manifest),
                    input_profile=L2_PROFILE, variables={'landsat_l2': landsat_l2_spec()})
    with locked(args.work_dir / 'run.lock', blocking=False):
        cache = Checkpoint(args.work_dir / 'checkpoint.sqlite3', contract)
        try:
            logging.info('Collecting %d/%d samples', len(rows), len(table))
            if not args.publish_only:
                download(rows, cache, args, fetcher=fetch_sample)
            report = publish(rows, cache, args, contract)
            print(json.dumps(report, indent=2), flush=True)
        finally:
            cache.close()


if __name__ == '__main__':
    main()
