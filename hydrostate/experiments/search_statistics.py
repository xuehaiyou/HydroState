"""Compute training-only scales for an audited architecture-search subset."""

import argparse
import copy
import json
from pathlib import Path

from mmengine.config import Config
from mmengine.registry import init_default_scope

from hydrostate.cli import configure_data_contract
from hydrostate.data.retrieval.statistics import compute_retrieval_statistics
from hydrostate.data.site.statistics import cohort_statistics
from hydrostate.registry import DATASETS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--selection-root', required=True)
    parser.add_argument('--fraction', type=float, default=.1)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    cfg = Config.fromfile(args.config)
    configure_data_contract(cfg)
    init_default_scope('hydrostate')
    cfg.train_dataloader.dataset.update(selection_root=args.selection_root,
                                        train_fraction=args.fraction)
    joint = DATASETS.build(dict(cfg.train_dataloader.dataset))
    product_cfg = copy.deepcopy(cfg)
    options = product_cfg.train_dataloader.dataset
    for key in ('site_split_root', 'selection_root', 'train_fraction', 'validation_fraction'):
        options.pop(key, None)
    options.update(type='RetrievalDataset', manifest=joint.product.manifest,
                   data_root=str(joint.product.root))
    products = compute_retrieval_statistics(product_cfg, num_workers=args.workers)
    sites = cohort_statistics([dict(manifest=site.manifest, data_root=str(site.root))
                               for site in joint.sites])
    joint.validate_site_statistics(sites)
    for name, value in [('target_stats.json', products), ('site_target_stats.json', sites)]:
        (output / name).write_text(json.dumps(value, indent=2))
    (output / '_SUCCESS.json').write_text(json.dumps(dict(
        selection_root=str(Path(args.selection_root).resolve()), fraction=args.fraction,
        training_windows=len(joint), product_windows=len(joint.product)), indent=2))


if __name__ == '__main__':
    main()
