"""Durable export receipts; cleanup is permitted only after cache commit."""

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path


class ExportState:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("""CREATE TABLE IF NOT EXISTS exports (
            id TEXT PRIMARY KEY, payload TEXT NOT NULL, status TEXT NOT NULL,
            task_id TEXT, files TEXT, attempts INTEGER NOT NULL DEFAULT 0,
            updated TEXT NOT NULL, error TEXT, part_index INTEGER NOT NULL DEFAULT 0)""")
        if "part_index" not in {r[1] for r in self.db.execute("PRAGMA table_info(exports)")}:
            self.db.execute("ALTER TABLE exports ADD COLUMN part_index INTEGER NOT NULL DEFAULT 0")
        self.db.commit()

    def add(self, job):
        payload = json.dumps(job, sort_keys=True)
        self.db.execute(
            "INSERT OR IGNORE INTO exports(id,payload,status,updated) VALUES(?,?,?,?)",
            (job["id"], payload, "planned", datetime.now(UTC).isoformat()),
        )
        self.db.commit()
        if self.get(job["id"])["payload"] != job:
            raise ValueError("Export ID collision or changed request contract")

    def get(self, job_id):
        row = self.db.execute("SELECT * FROM exports WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        result = dict(row)
        result["payload"] = json.loads(result["payload"])
        result["files"] = json.loads(result["files"]) if result["files"] else []
        return result

    def pending(self):
        ids = self.db.execute("SELECT id FROM exports WHERE status!='cleaned' ORDER BY rowid")
        return [self.get(row[0]) for row in ids.fetchall()]

    def update(self, job_id, **fields):
        if not fields or not set(fields) <= {
            "status", "task_id", "files", "attempts", "error", "part_index"
        }:
            raise ValueError("Unsupported export state fields")
        if fields.get("status") == "cleaned" and self.get(job_id)["status"] != "committed":
            raise RuntimeError("Remote deletion requires a committed cache receipt")
        if "files" in fields:
            fields["files"] = json.dumps(fields["files"], sort_keys=True)
        fields["updated"] = datetime.now(UTC).isoformat()
        self.db.execute(
            "UPDATE exports SET " + ",".join(f"{key}=?" for key in fields) + " WHERE id=?",
            [*fields.values(), job_id],
        )
        self.db.commit()

    def close(self):
        self.db.close()
