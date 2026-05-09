"""Memory store — SQLite persistent agent memory with priority and staleness."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any


class MemoryStore:
    """Persistent agent memory with LIKE search and priority ordering.

    Scopes: 'memory' (agent notes), 'user' (user profile).
    """

    def __init__(self, db_path: str | Path | None = None):
        self._db_path = Path(db_path or Path.home() / ".kairos" / "memory.db")
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self._db_path))
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
        return self._conn

    def _init_db(self) -> None:
        self._get_conn().executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scope TEXT NOT NULL DEFAULT 'memory',
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                priority REAL DEFAULT 0.5,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                UNIQUE(scope, key)
            );
            CREATE INDEX IF NOT EXISTS idx_mem_scope ON memories(scope);
            CREATE INDEX IF NOT EXISTS idx_mem_priority ON memories(priority DESC);
        """)
        self._get_conn().commit()

    def add(self, scope: str, key: str, value: str, priority: float = 0.5) -> str:
        now = time.time()
        conn = self._get_conn()
        existing = conn.execute(
            "SELECT id FROM memories WHERE scope=? AND key=?", (scope, key)
        ).fetchone()
        if existing:
            conn.execute(
                "UPDATE memories SET value=?, priority=?, updated_at=? WHERE id=?",
                (value, priority, now, existing[0]),
            )
            conn.commit()
            return "updated"
        conn.execute(
            "INSERT INTO memories (scope,key,value,priority,created_at,updated_at) VALUES (?,?,?,?,?,?)",
            (scope, key, value, priority, now, now),
        )
        conn.commit()
        return "added"

    def get(self, scope: str, key: str) -> str | None:
        row = self._get_conn().execute(
            "SELECT value FROM memories WHERE scope=? AND key=?", (scope, key)
        ).fetchone()
        return row["value"] if row else None

    def remove(self, scope: str, key: str) -> bool:
        conn = self._get_conn()
        c = conn.execute("DELETE FROM memories WHERE scope=? AND key=?", (scope, key))
        conn.commit()
        return c.rowcount > 0

    def search(self, query: str, scope: str | None = None, limit: int = 20) -> list[dict]:
        conn = self._get_conn()
        cond = ["(key LIKE ? OR value LIKE ?)"]
        params: list = [f"%{query}%", f"%{query}%"]
        if scope:
            cond.append("scope = ?")
            params.append(scope)
        sql = f"SELECT * FROM memories WHERE {' AND '.join(cond)} ORDER BY priority DESC, updated_at DESC LIMIT ?"
        params.append(limit)
        return [dict(r) for r in conn.execute(sql, params).fetchall()]

    def all(self, scope: str | None = None, min_priority: float = 0.0, max_age_days: float | None = None) -> list[dict]:
        cond = ["1=1"]
        params: list = []
        if scope:
            cond.append("scope = ?"); params.append(scope)
        if min_priority > 0:
            cond.append("priority >= ?"); params.append(min_priority)
        if max_age_days:
            params.append(time.time() - max_age_days * 86400)
            cond.append("updated_at >= ?")
        sql = f"SELECT * FROM memories WHERE {' AND '.join(cond)} ORDER BY priority DESC, updated_at DESC"
        return [dict(r) for r in self._get_conn().execute(sql, params).fetchall()]

    def clear(self, scope: str | None = None) -> int:
        conn = self._get_conn()
        c = conn.execute("DELETE FROM memories" if scope is None else "DELETE FROM memories WHERE scope=?", (scope,) if scope else ())
        conn.commit()
        return c.rowcount

    def count(self, scope: str | None = None) -> int:
        sql = "SELECT COUNT(*) FROM memories" if scope is None else "SELECT COUNT(*) FROM memories WHERE scope=?"
        row = self._get_conn().execute(sql, (scope,) if scope else ()).fetchone()
        return row[0] if row else 0

    def format_for_prompt(self, scope: str | None = None, max_chars: int = 3000) -> str:
        memories = self.all(scope=scope, min_priority=0.0, max_age_days=90)
        if not memories:
            return ""
        blocks: dict[str, list[str]] = {}
        total = 0
        for m in memories:
            entry = f"- **{m['key']}**: {m['value']}"
            if total + len(entry) > max_chars:
                break
            blocks.setdefault(m["scope"], []).append(entry)
            total += len(entry)
        return "\n\n".join(f"## {s.upper()}\n" + "\n".join(e) for s, e in blocks.items())

    def close(self) -> None:
        if self._conn:
            self._conn.close(); self._conn = None
