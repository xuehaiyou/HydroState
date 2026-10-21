"""Submission safety: missing data, successful preparation, and restart deduplication."""
import json
from types import SimpleNamespace

import pytest

from tools import watch_mtlora_collection as watch


def setup_record(tmp_path):
    data = tmp_path / 'data'
    data.mkdir()
    (tmp_path / 'provenance').mkdir()
    path = tmp_path / 'submission.json'
    record = dict(data_root=str(data), code_root=str(tmp_path), jobs={}, environment={})
    watch.save(path, record)
    return path, record, data


def test_incomplete_collection_never_submits(tmp_path, monkeypatch):
    path, record, _ = setup_record(tmp_path)
    monkeypatch.setattr(watch.subprocess, 'run', lambda *a, **k: pytest.fail('Unexpected sbatch'))
    assert not watch.advance(path, record)
    assert record['status'] == 'waiting_for_collection'
    assert record['jobs'] == {}


def test_preparation_gate_and_restart_no_duplicates(tmp_path, monkeypatch):
    path, record, data = setup_record(tmp_path)
    (data / '_COLLECTION_SUCCESS.json').write_text('{}')
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        assert json.loads(path.read_text())['jobs'][('prepare', 'mtlora', 'lora')[len(calls)-1]][
            'status'] == 'submitting'
        return SimpleNamespace(returncode=0, stdout=f'{100 + len(calls)}\n', stderr='',
                               check_returncode=lambda: None)

    monkeypatch.setattr(watch.subprocess, 'run', run)
    monkeypatch.setattr(watch.subprocess, 'check_output', lambda *a, **k: '101|RUNNING|0:0\n')
    assert not watch.advance(path, record)
    assert len(calls) == 1
    monkeypatch.setattr(watch.subprocess, 'check_output', lambda *a, **k: '101|COMPLETED|0:0\n')
    (tmp_path / 'provenance/ready.json').write_text('{}')
    monkeypatch.setattr(watch, 'free_gpu_slots', lambda: 0)
    assert not watch.advance(path, record)
    assert len(calls) == 1
    monkeypatch.setattr(watch, 'free_gpu_slots', lambda: 2)
    assert watch.advance(path, record)
    assert len(calls) == 3
    assert watch.advance(path, json.loads(path.read_text()))
    assert len(calls) == 3


def test_failed_or_ambiguous_submission_stops(tmp_path, monkeypatch):
    path, record, data = setup_record(tmp_path)
    (data / '_COLLECTION_SUCCESS.json').write_text('{}')
    record['jobs']['prepare'] = dict(job_id='101')
    monkeypatch.setattr(watch.subprocess, 'check_output', lambda *a, **k: '101|FAILED|1:0\n')
    with pytest.raises(RuntimeError, match='FAILED'):
        watch.advance(path, record)
    record['jobs']['prepare'] = dict(status='submitting')
    with pytest.raises(RuntimeError, match='ambiguous'):
        watch.advance(path, record)
