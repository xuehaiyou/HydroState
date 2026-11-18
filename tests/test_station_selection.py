import copy

import pytest
from mmengine.config import Config
from retrieval.test_model import statistics

from hydrostate.evaluation.joint_metrics import JointHydroStateMetric


def test_product_errors_cannot_change_station_selection_and_both_tasks_required():
    stats = dict(version='site_training_statistics_v1', split='train', ddof=0,
                 training_fingerprint='a' * 64, observations={
                     'site_sm': dict(count=10, mean=.2, std=.1),
                     'site_et': dict(count=10, mean=2., std=1.)})
    metric = JointHydroStateMetric(statistics(), stats,
                                  Config.fromfile('configs/hydrostate_joint.py').tasks,
                                  selection='site')
    rows = []
    for source, task, error in [('product', 'soil_moisture', 1.),
                                ('site', 'site_sm', .05), ('site', 'site_et', 1.)]:
        rows.append(dict(source=source, spatial_group_id='s', frequency_bin='0', **{
            f'{task}/count': 2, f'{task}/sum_abs': 2 * error,
            f'{task}/sum_sq': 2 * error**2, f'{task}/sum_err': 2 * error,
            f'{task}/sum_y': 3., f'{task}/sum_y2': 5.,
            f'{task}/sum_x': 3 + 2 * error,
            f'{task}/sum_x2': 5 + 6 * error + 2 * error**2,
            f'{task}/sum_xy': 5 + 3 * error}))
    first = metric.compute_metrics(rows)
    changed = copy.deepcopy(rows)
    changed[0]['soil_moisture/sum_sq'] = 10000.
    assert metric.compute_metrics(changed)['site_score'] == first['site_score']
    assert first['site_score'] == pytest.approx(-.75)
    with pytest.raises(ValueError, match='both SM and ET'):
        metric.compute_metrics(rows[:-1])
