"""Memory store — durable key-value storage for cross-session agent memory."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any


class MemoryStore:
    """Persistent memory for the agent.

    Stores compact, durable facts that survive across sessions:
      - User preferences and corrections
      - Environment details and conventions
      - Project facts and tool quirks

    Does NOT store: task progress, session outcomes, temporary state.

    Usage:
        store = MemoryStore()
        store.add("user", "prefers_concise_responses", "User likes short replies")
        results = store.search("concise")
    """

    def __init__(self, db_path: str | Path | None = None):
        self._db_path = Path(db_path or Path.home() / ".kairos" / "memory.db")
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scope TEXT NOT NULL DEFAULT 'memory',
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    UNIQUE(scope, key)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_scope
                ON memories(scope)
            """)
            conn.commit()

    def search(self, query: str, scope: str | None = None) -> list[dict[str, Any]]:
        """Search memories via full-text search on key and value fields."""
        with sqlite3.connect(str(self._db_path)) as conn:
            where_parts = ["(key LIKE ? OR value LIKE ?)"]
            params: list[Any] = [f"%{query}%", f"%{query}%"]
            if scope:
                where_parts.append("scope = ?")
                params.append(scope)
            where = " AND ".join(where_parts)
            sql = f"SELECT id, scope, key, value, created_at, updated_at FROM memories WHERE {where} ORDER BY updated_at DESC"
            rows = conn.execute(sql, params).fetchall()
            return [
                {
                    "id": r[0],
                    "scope": r[1],
                    "key": r[2],
                    "value": r[3],
                    "created_at": r[4],
                    "updated_at": r[5],
                }
                for r in rows
            ]

    def add(self, scope: str, key: str, value: str) -> str:
        """Add or update a memory entry. Returns 'added' or 'updated'."""
        now = time.time()
        with sqlite3.connect(str(self._db_path)) as conn:
            existing = conn.execute(
                "SELECT id FROM memories WHERE scope=? AND key=?",
                (scope, key),
            ).fetchone()

            if existing:
                conn.execute(
                    "UPDATE memories SET value=?, updated_at=? WHERE id=?",
                    (value, now, existing[0]),
                )
                conn.commit()
                return "updated"
            else:
                conn.execute(
                    "INSERT INTO memories (scope, key, value, created_at, updated_at) VALUES (?,?,?,?,?)",
                    (scope, key, value, now, now),
                )
                conn.commit()
                return "added"

    def get(self, scope: str, key: str) -> str | None:
        """Get a memory value by scope and key."""
        with sqlite3.connect(str(self._db_path)) as conn:
            row = conn.execute(
                "SELECT value FROM memories WHERE scope=? AND key=?",
                (scope, key),
            ).fetchone()
            return row[0] if row else None

    def all(self, scope: str | None = None) -> list[dict[str, Any]]:
        """Get all memories, optionally filtered by scope."""
        with sqlite3.connect(str(self._db_path)) as conn:
            if scope:
                rows = conn.execute(
                    "SELECT id, scope, key, value, created_at, updated_at FROM memories WHERE scope=? ORDER BY key",
                    (scope,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, scope, key, value, created_at, updated_at FROM memories ORDER BY scope, key",
                ).fetchall()

            return [
                {
                    "id": r[0],
                    "scope": r[1],
                    "key": r[2],
                    "value": r[3],
                    "created_at": r[4],
                    "updated_at": r[5],
                }
                for r in rows
            ]

    def remove(self, scope: str, key: str) -> bool:
        """Remove a memory entry. Returns True if found and removed."""
        with sqlite3.connect(str(self._db_path)) as conn:
            cursor = conn.execute(
                "DELETE FROM memories WHERE scope=? AND key=?",
                (scope, key),
            )
            conn.commit()
            return cursor.rowcount > 0

    def clear(self, scope: str | None = None) -> int:
        """Clear memories. If scope is None, clears all. Returns count removed."""
        with sqlite3.connect(str(self._db_path)) as conn:
            if scope:
                cursor = conn.execute(
                    "DELETE FROM memories WHERE scope=?",
                    (scope,),
                )
            else:
                cursor = conn.execute("DELETE FROM memories")
            conn.commit()
            return cursor.rowcount

    def format_for_prompt(self, scope: str | None = None) -> str:
        """Format memories as a system prompt injection block."""
        memories = self.all(scope=scope)
        if not memories:
            return ""

        blocks: dict[str, list[str]] = {}
        for m in memories:
            blocks.setdefault(m["scope"], []).append(f"- **{m['key']}**: {m['value']}")

        parts = []
        for scope_name, entries in blocks.items():
            parts.append(f"## {scope_name.upper()}\n" + "\n".join(entries))

        return "\n\n".join(parts)

    def count(self, scope: str | None = None) -> int:
        """Count memories, optionally filtered by scope."""
        with sqlite3.connect(str(self._db_path)) as conn:
            if scope:
                row = conn.execute(
                    "SELECT COUNT(*) FROM memories WHERE scope=?",
                    (scope,),
                ).fetchone()
            else:
                row = conn.execute("SELECT COUNT(*) FROM memories").fetchone()
            return row[0] if row else 0
