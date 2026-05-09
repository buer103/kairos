"""Memory backends — abstract interface + SQLite (FTS5) + in-memory dict implementations.

Provides:
- MemoryBackend ABC: contract for all memory storage
- SQLiteBackend: SQLite + FTS5 full-text search with BM25 ranking
- DictBackend: lightweight in-memory dict for testing / throwaway sessions

Thread-safe. Python 3.12+.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


# ── Abstract Backend ────────────────────────────────────────────────


class MemoryBackend(ABC):
    """Abstract contract for memory storage backends.

    All methods are thread-safe in concrete implementations.
    """

    @abstractmethod
    def save(
        self,
        key: str,
        value: str,
        *,
        category: str = "",
        metadata: dict[str, Any] | None = None,
        ttl: float | None = None,
    ) -> None:
        """Save / upsert a memory entry.

        If *key* already exists the value, category, metadata and TTL are
        updated in-place.

        Args:
            key: Unique identifier for this memory.
            value: The text content.
            category: Tag for grouping (``"preference"``, ``"fact"``,
                      ``"conversation"``, ``"project"``, or custom).
            metadata: Arbitrary JSON-serialisable extra data.
            ttl: Time-to-live in seconds from now.  Expired entries are
                 silently pruned on the next read / write operation.
        """
        ...

    @abstractmethod
    def load(self, key: str) -> dict[str, Any] | None:
        """Load a single memory entry by key.  Returns *None* if not found."""
        ...

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete a memory entry.  Returns ``True`` if a row was removed."""
        ...

    @abstractmethod
    def search(
        self, query: str, *, limit: int = 20, category: str | None = None
    ) -> list[dict[str, Any]]:
        """Full-text search across *key*, *value*, and *category* fields.

        Results are ranked by relevance (BM25 for SQLiteBackend).

        Args:
            query: Free-text search term(s).
            limit: Maximum results to return.
            category: Optional category filter.
        """
        ...

    @abstractmethod
    def list_keys(
        self, prefix: str = "", *, category: str | None = None
    ) -> list[str]:
        """List all keys, optionally filtered by prefix and/or category."""
        ...

    @abstractmethod
    def clear(self, *, category: str | None = None) -> int:
        """Remove all (or category-filtered) memories.  Returns count removed."""
        ...

    # ── Convenience helpers (derived, override if cheaper) ──────────

    def exists(self, key: str) -> bool:
        return self.load(key) is not None


# ── SQLite + FTS5 Backend ────────────────────────────────────────────


class SQLiteBackend(MemoryBackend):
    """SQLite-backed memory with FTS5 full-text search and BM25 ranking.

    Database path defaults to ``~/.kairos/memory/memory.db``.

    Schema::

        memory (key, value, category, created_at, updated_at, metadata, expires_at)
        memory_fts USING fts5(key, value, category, content=memory)

    Triggers keep the FTS index in sync on INSERT / UPDATE / DELETE.
    All public methods acquire a ``threading.Lock``; suitable for
    multi-threaded use.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._db_path = Path(
            db_path or Path.home() / ".kairos" / "memory" / "memory.db"
        )
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.Lock()
        self._init_db()

    # ── connection management ──────────────────────────────────────

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
        return self._conn

    def _init_db(self) -> None:
        with self._lock:
            conn = self._get_conn()
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS memory (
                    key         TEXT PRIMARY KEY,
                    value       TEXT NOT NULL,
                    category    TEXT DEFAULT '',
                    created_at  REAL NOT NULL,
                    updated_at  REAL NOT NULL,
                    metadata    TEXT DEFAULT '{}',
                    expires_at  REAL
                );

                CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
                    key, value, category,
                    content=memory, content_rowid=rowid
                );

                CREATE TRIGGER IF NOT EXISTS memory_ai AFTER INSERT ON memory BEGIN
                    INSERT INTO memory_fts(rowid, key, value, category)
                    VALUES (new.rowid, new.key, new.value, new.category);
                END;

                CREATE TRIGGER IF NOT EXISTS memory_ad AFTER DELETE ON memory BEGIN
                    INSERT INTO memory_fts(memory_fts, rowid, key, value, category)
                    VALUES ('delete', old.rowid, old.key, old.value, old.category);
                END;

                CREATE TRIGGER IF NOT EXISTS memory_au AFTER UPDATE ON memory BEGIN
                    INSERT INTO memory_fts(memory_fts, rowid, key, value, category)
                    VALUES ('delete', old.rowid, old.key, old.value, old.category);
                    INSERT INTO memory_fts(rowid, key, value, category)
                    VALUES (new.rowid, new.key, new.value, new.category);
                END;
            """)
            conn.commit()

    # ── TTL cleanup ─────────────────────────────────────────────────

    def _expire(self) -> None:
        """Remove entries whose *expires_at* has passed."""
        now = time.time()
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                "DELETE FROM memory WHERE expires_at IS NOT NULL AND expires_at <= ?",
                (now,),
            )
            conn.commit()

    # ── Backend API ─────────────────────────────────────────────────

    def save(
        self,
        key: str,
        value: str,
        *,
        category: str = "",
        metadata: dict[str, Any] | None = None,
        ttl: float | None = None,
    ) -> None:
        self._expire()
        now = time.time()
        expires_at = (now + ttl) if ttl is not None else None
        meta_json = json.dumps(metadata or {}, ensure_ascii=False)

        with self._lock:
            conn = self._get_conn()
            existing = conn.execute(
                "SELECT rowid FROM memory WHERE key = ?", (key,)
            ).fetchone()
            if existing:
                conn.execute(
                    """UPDATE memory
                       SET value=?, category=?, updated_at=?, metadata=?, expires_at=?
                       WHERE key=?""",
                    (value, category, now, meta_json, expires_at, key),
                )
            else:
                conn.execute(
                    """INSERT INTO memory
                       (key, value, category, created_at, updated_at, metadata, expires_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (key, value, category, now, now, meta_json, expires_at),
                )
            conn.commit()

    def load(self, key: str) -> dict[str, Any] | None:
        self._expire()
        with self._lock:
            row = self._get_conn().execute(
                "SELECT * FROM memory WHERE key = ?", (key,)
            ).fetchone()
        if row is None:
            return None
        return dict(row)

    def delete(self, key: str) -> bool:
        with self._lock:
            conn = self._get_conn()
            c = conn.execute("DELETE FROM memory WHERE key = ?", (key,))
            conn.commit()
            return c.rowcount > 0

    def search(
        self, query: str, *, limit: int = 20, category: str | None = None
    ) -> list[dict[str, Any]]:
        self._expire()
        # Escape FTS5 double-quote special character.
        safe = query.replace('"', '""')
        fts_query = f'"{safe}"'

        with self._lock:
            conn = self._get_conn()
            match category:
                case None:
                    rows = conn.execute(
                        """SELECT m.*, bm25(memory_fts, 0, 0, 0, 0) AS rank
                           FROM memory m
                           JOIN memory_fts ON m.rowid = memory_fts.rowid
                           WHERE memory_fts MATCH ?
                           ORDER BY rank
                           LIMIT ?""",
                        (fts_query, limit),
                    ).fetchall()
                case _:
                    rows = conn.execute(
                        """SELECT m.*, bm25(memory_fts, 0, 0, 0, 0) AS rank
                           FROM memory m
                           JOIN memory_fts ON m.rowid = memory_fts.rowid
                           WHERE memory_fts MATCH ? AND m.category = ?
                           ORDER BY rank
                           LIMIT ?""",
                        (fts_query, category, limit),
                    ).fetchall()
        return [dict(r) for r in rows]

    def list_keys(
        self, prefix: str = "", *, category: str | None = None
    ) -> list[str]:
        self._expire()
        with self._lock:
            conn = self._get_conn()
            match (bool(prefix), category):
                case (True, None):
                    rows = conn.execute(
                        "SELECT key FROM memory WHERE key LIKE ? ORDER BY key",
                        (f"{prefix}%",),
                    ).fetchall()
                case (True, _):
                    rows = conn.execute(
                        "SELECT key FROM memory WHERE key LIKE ? AND category = ? ORDER BY key",
                        (f"{prefix}%", category),
                    ).fetchall()
                case (False, None):
                    rows = conn.execute(
                        "SELECT key FROM memory ORDER BY key"
                    ).fetchall()
                case (False, _):
                    rows = conn.execute(
                        "SELECT key FROM memory WHERE category = ? ORDER BY key",
                        (category,),
                    ).fetchall()
        return [r["key"] for r in rows]

    def clear(self, *, category: str | None = None) -> int:
        with self._lock:
            conn = self._get_conn()
            match category:
                case None:
                    c = conn.execute("DELETE FROM memory")
                case _:
                    c = conn.execute("DELETE FROM memory WHERE category = ?", (category,))
            conn.commit()
            return c.rowcount

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None


# ── In-Memory Dict Backend ───────────────────────────────────────────


class DictBackend(MemoryBackend):
    """Lightweight in-memory backend backed by a plain :class:`dict`.

    Suitable for testing, throwaway sessions, or environments where
    persistence is not desired.
    """

    def __init__(self) -> None:
        self._store: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    # ── Backend API ─────────────────────────────────────────────────

    def save(
        self,
        key: str,
        value: str,
        *,
        category: str = "",
        metadata: dict[str, Any] | None = None,
        ttl: float | None = None,
    ) -> None:
        now = time.time()
        with self._lock:
            created = (
                self._store[key]["created_at"]
                if key in self._store
                else now
            )
            self._store[key] = {
                "key": key,
                "value": value,
                "category": category,
                "created_at": created,
                "updated_at": now,
                "metadata": json.dumps(metadata or {}, ensure_ascii=False),
                "expires_at": (now + ttl) if ttl is not None else None,
            }

    def load(self, key: str) -> dict[str, Any] | None:
        self._expire()
        with self._lock:
            entry = self._store.get(key)
        return dict(entry) if entry is not None else None

    def delete(self, key: str) -> bool:
        with self._lock:
            return self._store.pop(key, None) is not None

    def search(
        self, query: str, *, limit: int = 20, category: str | None = None
    ) -> list[dict[str, Any]]:
        self._expire()
        q = query.lower()
        results: list[dict[str, Any]] = []
        with self._lock:
            for entry in self._store.values():
                if category is not None and entry["category"] != category:
                    continue
                if q in entry["key"].lower() or q in entry["value"].lower():
                    results.append(dict(entry))
        results.sort(key=lambda e: e["updated_at"], reverse=True)
        return results[:limit]

    def list_keys(
        self, prefix: str = "", *, category: str | None = None
    ) -> list[str]:
        self._expire()
        keys: list[str] = []
        with self._lock:
            for k, entry in self._store.items():
                if prefix and not k.startswith(prefix):
                    continue
                if category is not None and entry["category"] != category:
                    continue
                keys.append(k)
        return sorted(keys)

    def clear(self, *, category: str | None = None) -> int:
        with self._lock:
            match category:
                case None:
                    count = len(self._store)
                    self._store.clear()
                    return count
                case _:
                    doomed = [k for k, v in self._store.items() if v["category"] == category]
                    for k in doomed:
                        del self._store[k]
                    return len(doomed)

    # ── Internal ────────────────────────────────────────────────────

    def _expire(self) -> None:
        now = time.time()
        with self._lock:
            expired = [
                k
                for k, v in self._store.items()
                if v.get("expires_at") is not None and v["expires_at"] <= now
            ]
            for k in expired:
                del self._store[k]
