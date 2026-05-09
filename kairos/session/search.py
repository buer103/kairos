"""Session search — FTS5 full-text search across historical sessions."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any


class SessionSearch:
    """Full-text search across agent session transcripts.

    Uses SQLite FTS5 for fast keyword search.
    Indexes session messages, tool calls, and metadata.
    """

    def __init__(self, db_path: str | Path | None = None):
        self._db_path = Path(db_path or Path.home() / ".kairos" / "sessions.db")
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(str(self._db_path)) as conn:
            # Sessions table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    title TEXT DEFAULT '',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    message_count INTEGER DEFAULT 0,
                    tool_call_count INTEGER DEFAULT 0,
                    metadata TEXT DEFAULT '{}'
                )
            """)

            # Messages for FTS indexing (no FK constraint to simplify deletes)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS session_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    tool_name TEXT
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_session_msgs_sid
                ON session_messages(session_id)
            """)

            # FTS5 — drop and recreate to avoid stale triggers
            conn.execute("DROP TABLE IF EXISTS session_messages_fts")
            conn.execute("""
                CREATE VIRTUAL TABLE session_messages_fts USING fts5(
                    role, content, tool_name
                )
            """)

            conn.commit()

    def index_session(
        self,
        session_id: str,
        messages: list[dict],
        metadata: dict | None = None,
        title: str = "",
    ) -> None:
        """Index a session's messages for search.

        Args:
            session_id: unique session identifier
            messages: list of message dicts with role and content
            metadata: optional session-level metadata
            title: optional session title
        """
        now = time.time()
        tool_count = sum(1 for m in messages if m.get("tool_calls") or m.get("role") == "tool")

        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute("PRAGMA foreign_keys = ON")

            # Upsert session
            conn.execute(
                """INSERT OR REPLACE INTO sessions (id, title, created_at, updated_at, message_count, tool_call_count, metadata)
                   VALUES (?, ?, COALESCE((SELECT created_at FROM sessions WHERE id=?), ?), ?, ?, ?, ?)""",
                (session_id, title, session_id, now, now, len(messages), tool_count,
                 json.dumps(metadata or {}, ensure_ascii=False)),
            )

            # Remove old messages for this session
            conn.execute("DELETE FROM session_messages WHERE session_id=?", (session_id,))

            # Insert messages and FTS entries manually
            for m in messages:
                role = m.get("role", "unknown")
                content = m.get("content", "")
                if isinstance(content, (dict, list)):
                    content = json.dumps(content, ensure_ascii=False)

                tool_name = None
                if m.get("tool_calls"):
                    tool_name = ",".join(
                        tc.get("function", {}).get("name", "unknown")
                        for tc in m["tool_calls"]
                    )

                cursor = conn.execute(
                    "INSERT INTO session_messages (session_id, role, content, tool_name) VALUES (?,?,?,?)",
                    (session_id, role, content[:10000], tool_name),
                )
                msg_id = cursor.lastrowid

                # Manual FTS insert
                conn.execute(
                    "INSERT INTO session_messages_fts(rowid, role, content, tool_name) VALUES (?,?,?,?)",
                    (msg_id, role, content[:10000], tool_name or ""),
                )

            conn.commit()

    def search(
        self,
        query: str,
        limit: int = 10,
        date_from: float | None = None,
        date_to: float | None = None,
        role: str | None = None,
        tool: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search sessions by keyword with optional filters.

        Args:
            query: FTS5 search query (supports boolean operators)
            limit: max results
            date_from: filter sessions after this timestamp
            date_to: filter sessions before this timestamp
            role: filter messages by role (user/assistant/tool/system)
            tool: filter messages by tool name used
        """
        conditions = ["session_messages_fts MATCH ?"]
        params: list[Any] = [query]

        if role:
            conditions.append("sm.role = ?")
            params.append(role)

        if tool:
            conditions.append("sm.tool_name LIKE ?")
            params.append(f"%{tool}%")

        where = " AND ".join(conditions)

        with sqlite3.connect(str(self._db_path)) as conn:
            sql = f"""
                SELECT DISTINCT s.id, s.title, s.created_at, s.message_count, s.tool_call_count,
                       sm.role, sm.content, sm.tool_name,
                       snippet(session_messages_fts, 2, '<b>', '</b>', '...', 32) as snippet
                FROM session_messages_fts
                INNER JOIN session_messages sm ON session_messages_fts.rowid = sm.id
                INNER JOIN sessions s ON sm.session_id = s.id
                WHERE {where}
                ORDER BY rank
                LIMIT ?
            """
            params.append(limit)
            rows = conn.execute(sql, params).fetchall()

            results = []
            for r in rows:
                session_id = r[0]
                created_at = r[2]

                # Apply date filters in Python (timestamps are stored as REAL)
                if date_from and created_at < date_from:
                    continue
                if date_to and created_at > date_to:
                    continue

                results.append({
                    "session_id": session_id,
                    "title": r[1],
                    "created_at": created_at,
                    "message_count": r[3],
                    "tool_call_count": r[4],
                    "matched_role": r[5],
                    "matched_content": r[6][:500],
                    "matched_tool": r[7],
                    "snippet": r[8],
                })

            return results

    def search_sessions(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Search session-level metadata (titles, summaries) without message-level detail."""
        with sqlite3.connect(str(self._db_path)) as conn:
            sql = """
                SELECT DISTINCT s.id, s.title, s.created_at, s.message_count, s.tool_call_count, s.metadata
                FROM sessions s
                INNER JOIN session_messages_fts fts ON s.id = (
                    SELECT sm.session_id FROM session_messages sm
                    WHERE sm.id = fts.rowid
                )
                WHERE session_messages_fts MATCH ?
                ORDER BY rank
                LIMIT ?
            """
            rows = conn.execute(sql, (query, limit)).fetchall()
            return [
                {
                    "session_id": r[0],
                    "title": r[1],
                    "created_at": r[2],
                    "message_count": r[3],
                    "tool_call_count": r[4],
                    "metadata": json.loads(r[5]) if r[5] else {},
                }
                for r in rows
            ]

    def recent_sessions(self, limit: int = 20) -> list[dict[str, Any]]:
        """List recent sessions by last update time."""
        with sqlite3.connect(str(self._db_path)) as conn:
            rows = conn.execute(
                """SELECT id, title, created_at, updated_at, message_count, tool_call_count, metadata
                   FROM sessions ORDER BY updated_at DESC LIMIT ?""",
                (limit,),
            ).fetchall()
            return [
                {
                    "session_id": r[0],
                    "title": r[1],
                    "created_at": r[2],
                    "updated_at": r[3],
                    "message_count": r[4],
                    "tool_call_count": r[5],
                    "metadata": json.loads(r[6]) if r[6] else {},
                }
                for r in rows
            ]

    def delete_session(self, session_id: str) -> bool:
        """Remove a session and its indexed messages."""
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute("DELETE FROM session_messages WHERE session_id=?", (session_id,))
            cursor = conn.execute("DELETE FROM sessions WHERE id=?", (session_id,))
            conn.commit()
            return cursor.rowcount > 0

    def count(self) -> int:
        """Total indexed sessions."""
        with sqlite3.connect(str(self._db_path)) as conn:
            row = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()
            return row[0] if row else 0
