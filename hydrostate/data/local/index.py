"""Transactional discovery and per-scene checkpoints for the frozen local catalog."""

import json
import sqlite3
from pathlib import Path


class CatalogCheckpoint:
    """Called under the regional prepare lock; a failed scene is never committed.

    Keep SQLite's rollback journal (not WAL) for the shared HPC filesystem.
    Published JSON remains the read-only interface consumed by preparation.
    """

    def __init__(self, path, contract):
        self.db = sqlite3.connect(path)
        try:
            self.db.execute('PRAGMA synchronous=FULL')
            self.db.execute('CREATE TABLE IF NOT EXISTS config (value TEXT NOT NULL)')
            self.db.execute('CREATE TABLE IF NOT EXISTS discoveries '
                            '(sensor TEXT PRIMARY KEY, paths TEXT NOT NULL)')
            self.db.execute('CREATE TABLE IF NOT EXISTS scenes '
                            '(sensor TEXT, path TEXT, result TEXT NOT NULL, '
                            'PRIMARY KEY(sensor, path))')
            encoded = json.dumps(dict(format_version=1, contract=contract), sort_keys=True)
            previous = self.db.execute('SELECT value FROM config').fetchone()
            if previous and previous[0] != encoded:
                raise ValueError('Local index checkpoint configuration changed; use a new root')
            if not previous:
                with self.db:
                    self.db.execute('INSERT INTO config VALUES (?)', (encoded,))
        except BaseException:
            self.db.close()
            raise

    def candidates(self, sensor, discover):
        saved = self.db.execute('SELECT paths FROM discoveries WHERE sensor=?',
                                (sensor,)).fetchone()
        if saved:
            return [Path(path) for path in json.loads(saved[0])]
        paths = sorted(set(discover()))
        with self.db:
            self.db.execute('INSERT INTO discoveries VALUES (?,?)',
                            (sensor, json.dumps([str(path) for path in paths])))
        return paths

    def get(self, sensor, path):
        saved = self.db.execute('SELECT result FROM scenes WHERE sensor=? AND path=?',
                                (sensor, str(path))).fetchone()
        if not saved:
            return None
        result = json.loads(saved[0])
        source = result.get('scene', result)
        for filename, expected in source['snapshot'].items():
            stat = Path(filename).stat()
            if [stat.st_size, stat.st_mtime_ns] != expected:
                raise ValueError(f'Indexed source changed: {filename}; use a new root')
        return result

    def save(self, sensor, path, result):
        with self.db:
            self.db.execute('INSERT INTO scenes VALUES (?,?,?)',
                            (sensor, str(path), json.dumps(result, allow_nan=False)))

    def close(self):
        self.db.close()
