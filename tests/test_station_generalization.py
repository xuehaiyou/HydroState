import numpy as np
import pandas as pd

from hydrostate.experiments.station_generalization import station_report


def test_between_and_within_can_have_opposite_signs():
    rows = []
    for station, level in enumerate([0., 10., 20.]):
        for i in range(3):
            for sensor in range(2):
                for mode in ('dynamic', 'local_sm'):
                    rows.append(dict(task='site_sm', site=station, station=station,
                                     date=f'2020-01-0{i+1}', sample_id=f'{station}-{i}',
                                     observation=sensor, pred=level-i, target=level+i, mode=mode))
    out, stations = station_report(pd.DataFrame(rows), draws=100)
    assert np.allclose(out[out.metric=='between_r'].value, 1.)
    assert np.allclose(out[out.metric=='within_r'].value, -1.)
    assert (out.windows==9).all() and (stations.windows==3).all()
    assert (out.change==0).all()
