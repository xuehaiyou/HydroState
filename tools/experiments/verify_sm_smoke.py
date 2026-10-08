#!/usr/bin/env python
"""Mark smoke passed only after all four losses run on two GPUs with paired init."""
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
reports = []
for config in (root/'screen/initial_configs.txt').read_text().splitlines():
    directory = root/'smoke'/Path(config).stem
    done = json.loads((directory/'completed.json').read_text())
    init = json.loads((directory/'initialization.json').read_text())
    if done['iterations'] != 2 or not done['smoke'] or init['world_size'] != 2:
        raise RuntimeError('Smoke incomplete')
    reports.append(init)
if len(reports) != 4 or any(r['initialization_sha256'] != reports[0]['initialization_sha256'] for r in reports):
    raise RuntimeError('Paired initialization mismatch')
(root/'smoke/passed.json').write_text(json.dumps(dict(passed=True,reports=reports),indent=2)+'\n')
print('All four DDP smoke runs passed with identical shared/SM initialization.')
