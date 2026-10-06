"""Armazenamento: metadados numa tabela SQLite e vetores numa tabela vec0 (sqlite-vec)."""

from __future__ import annotations

import sqlite3
from dataclasses import astuple, fields
from datetime import datetime

import sqlite_vec

from .base import Record

_COLS = [f.name for f in fields(Record)]
_DATES = {"created_at", "last_used_at"}
_BOOLS = {"active"}


def _to_row(rec: Record) -> tuple:
    out = []
    for name, v in zip(_COLS, astuple(rec)):
        if name in _DATES and v is not None:
            v = v.isoformat()
        elif name in _BOOLS:
            v = int(v)
        out.append(v)
    return tuple(out)


def _from_row(row: sqlite3.Row) -> Record:
    d = dict(row)
    for k in _DATES:
        if d[k] is not None:
            d[k] = datetime.fromisoformat(d[k])
    d["active"] = bool(d["active"])
    return Record(**d)


class VectorStore:
    def __init__(self, dim: int, path: str = ":memory:"):
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.enable_load_extension(True)
        sqlite_vec.load(self.db)
        self.db.enable_load_extension(False)
        self.db.executescript(f"""
            CREATE TABLE memories (
                id INTEGER PRIMARY KEY, content TEXT NOT NULL, kind TEXT NOT NULL,
                source TEXT NOT NULL, trust REAL NOT NULL, created_at TEXT NOT NULL,
                last_used_at TEXT, use_count INTEGER NOT NULL, subject_key TEXT,
                value TEXT, supersedes INTEGER, active INTEGER NOT NULL, tier TEXT NOT NULL,
                origin TEXT
            );
            CREATE INDEX memories_key ON memories(subject_key, active);
            CREATE VIRTUAL TABLE memory_vec USING vec0(embedding float[{dim}] distance_metric=cosine);
        """)

    def add(self, rec: Record, embedding: list[float]) -> Record:
        cols = [c for c in _COLS if c != "id"]
        row = _to_row(rec)[1:]
        cur = self.db.execute(
            f"INSERT INTO memories ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})", row
        )
        rec.id = cur.lastrowid
        self.db.execute(
            "INSERT INTO memory_vec(rowid, embedding) VALUES (?, ?)",
            (rec.id, sqlite_vec.serialize_float32(embedding)),
        )
        return rec

    def update(self, rec: Record) -> None:
        cols = [c for c in _COLS if c != "id"]
        self.db.execute(
            f"UPDATE memories SET {', '.join(f'{c} = ?' for c in cols)} WHERE id = ?",
            (*_to_row(rec)[1:], rec.id),
        )

    def get_many(self, ids: list[int]) -> list[Record]:
        if not ids:
            return []
        q = f"SELECT * FROM memories WHERE id IN ({', '.join('?' * len(ids))})"
        return [_from_row(r) for r in self.db.execute(q, ids)]

    def all(self) -> list[Record]:
        return [_from_row(r) for r in self.db.execute("SELECT * FROM memories ORDER BY id")]

    def active_by_key(self, key: str) -> Record | None:
        row = self.db.execute(
            "SELECT * FROM memories WHERE subject_key = ? AND active = 1 ORDER BY id DESC LIMIT 1",
            (key,),
        ).fetchone()
        return _from_row(row) if row else None

    def active_keys(self) -> list[str]:
        rows = self.db.execute(
            "SELECT DISTINCT subject_key FROM memories WHERE subject_key IS NOT NULL AND active = 1"
        )
        return sorted(r[0] for r in rows)

    def knn(self, embedding: list[float], k: int) -> list[tuple[Record, float]]:
        """k vizinhos mais próximos com similaridade de cosseno (1 - distância)."""
        rows = self.db.execute(
            "SELECT rowid, distance FROM memory_vec WHERE embedding MATCH ? AND k = ?",
            (sqlite_vec.serialize_float32(embedding), k),
        ).fetchall()
        sims = {r[0]: 1.0 - r[1] for r in rows}
        recs = {r.id: r for r in self.get_many(list(sims))}
        return [(recs[i], sims[i]) for i in sims]
