import json

from hydrostate.experiments.search_report import TASKS, best_validation


def test_report_keeps_one_epoch_and_ignores_incomplete_log_record(tmp_path):
    directory = tmp_path / '20261117/vis_data'
    directory.mkdir(parents=True)
    rows = []
    for epoch, score, error in ((2, -.8, .1), (4, -.5, .2), (6, -.7, .05)):
        row = dict(step=epoch, joint_score=score)
        for source, task in TASKS:
            row.update({f'{source}/{task}_{metric}': value for metric, value in
                        dict(r2=.6, rmse=error, mae=error / 2, bias=.01, count=100).items()})
        rows.append(json.dumps(row))
    (directory / 'scalars.json').write_text('\n'.join(rows) + '\n{"joint_score":')
    (tmp_path / 'best_joint_score_epoch_4.pth').touch()
    result = best_validation(tmp_path)
    assert result['epoch'] == 4 and result['completed_validation_epoch'] == 6
    assert result['checkpoint_present']
    assert all(values['rmse'] == .2 for values in result['metrics'].values())
