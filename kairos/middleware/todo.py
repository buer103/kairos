"""Todo middleware — persistent todo lists with compression integration.

DeerFlow layer 6 — registers as BeforeCompressionHook so todos survive truncation.

Enhancements:
  - SQLite-backed persistence
  - BeforeCompressionHook integration
  - Full CRUD (add/update/complete/delete/reorder)
  - Max items cap
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any

from kairos.core.middleware import Middleware

logger = logging.getLogger("kairos.middleware.todo")


class TodoMiddleware(Middleware):
    """Persists and recovers todo lists across context compression.

    Hook: before_model — injects truncated todos.
          Registered as BeforeCompressionHook — saves state before compression.
    """

    MAX_ITEMS = 20

    def __init__(self, db_path: str | None = None):
        self._todos: list[dict] = []
        self._injected = False
        self._db_path = db_path or str(Path.home() / ".kairos" / "todos.db")
        self._conn: sqlite3.Connection | None = None

    def _get_db(self) -> sqlite3.Connection:
        if self._conn is None:
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self._db_path)
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS todos ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT,"
                "content TEXT NOT NULL, status TEXT DEFAULT 'pending',"
                "created_at REAL, updated_at REAL)"
            )
            self._conn.commit()
        return self._conn

    # ── BeforeCompressionHook — save before compression ────────

    def __call__(self, messages_to_compress: list[dict], runtime: dict) -> None:
        """Called by ContextCompressor before messages are summarized away."""
        self._save_to_db()
        self._injected = False  # Allow re-injection after compression
        logger.debug("Saved %d todos before compression", len(self._todos))

    # ── before_model — recovery injection ──────────────────────

    def before_model(self, state: Any, runtime: dict[str, Any]) -> dict[str, Any] | None:
        if not runtime.get("is_plan_mode"):
            return None

        # Already in context?
        has_todos_in_context = self._check_todos_in_messages(state)
        if has_todos_in_context or self._injected:
            return None

        if not self._todos:
            return None

        # Inject
        reminder = self._format_reminder()
        state.messages.append({"role": "user", "content": reminder})
        self._injected = True
        return None

    # ── Public API ─────────────────────────────────────────────

    def set_todos(self, todos: list[dict]) -> None:
        self._todos = [
            {"content": t.get("content", ""), "status": t.get("status", "pending")}
            for t in todos[:self.MAX_ITEMS]
        ]
        self._injected = False

    def add(self, content: str, status: str = "pending") -> dict:
        if len(self._todos) >= self.MAX_ITEMS:
            return {"error": f"Max {self.MAX_ITEMS} items"}
        item = {"content": content, "status": status}
        self._todos.append(item)
        self._injected = False
        return item

    def update(self, index: int, content: str | None = None, status: str | None = None) -> dict | None:
        if 0 <= index < len(self._todos):
            if content is not None:
                self._todos[index]["content"] = content
            if status is not None:
                self._todos[index]["status"] = status
            self._injected = False
            return self._todos[index]
        return None

    def delete(self, index: int) -> bool:
        if 0 <= index < len(self._todos):
            self._todos.pop(index)
            self._injected = False
            return True
        return False

    def reorder(self, indices: list[int]) -> bool:
        if len(indices) != len(self._todos):
            return False
        self._todos = [self._todos[i] for i in indices]
        return True

    def get_todos(self) -> list[dict]:
        return list(self._todos)

    def clear(self) -> None:
        self._todos.clear()
        self._injected = False

    # ── Helpers ────────────────────────────────────────────────

    @staticmethod
    def _check_todos_in_messages(state: Any) -> bool:
        messages = getattr(state, "messages", [])
        return any(
            m.get("tool_calls") and any(
                tc.get("function", {}).get("name") == "write_todos"
                for tc in m.get("tool_calls", [])
            )
            for m in messages
        )

    def _format_reminder(self) -> str:
        items = "\n".join(
            f"- [{'x' if t.get('status') == 'completed' else ' '}] {t.get('content', '')}"
            for t in self._todos
        )
        return (
            f"<system_reminder>\n"
            f"Your todo list was truncated by context compression. Current state:\n"
            f"{items}\n"
            f"Continue working on pending items.\n"
            f"</system_reminder>"
        )

    def _save_to_db(self) -> None:
        try:
            db = self._get_db()
            db.execute("DELETE FROM todos")
            now = time.time()
            for t in self._todos:
                db.execute(
                    "INSERT INTO todos (content, status, created_at, updated_at) VALUES (?, ?, ?, ?)",
                    (t["content"], t.get("status", "pending"), now, now),
                )
            db.commit()
        except Exception as e:
            logger.debug("Failed to persist todos: %s", e)

    def __repr__(self) -> str:
        return f"TodoMiddleware(items={len(self._todos)})"
