"""Todo middleware — persists and recovers todo lists across context compression.

When the agent generates a todo list (via write_todos tool), and the context
is later compressed, the model loses track of it. This middleware detects the
loss and injects a reminder.

DeerFlow layer 6 — runs after Summarization, checking if todos were truncated.
"""

from __future__ import annotations

from typing import Any

from kairos.core.middleware import Middleware


class TodoMiddleware(Middleware):
    """Persists todo list state and recovers after context truncation.

    Hook: before_model — checks if the todo list is still in context,
    and injects a reminder if it was truncated away.

    Usage:
        Set `is_plan_mode=True` in runtime to enable.
    """

    def __init__(self):
        self._todos: list[dict[str, str]] = []
        self._todos_injected = False

    def before_model(self, state: Any, runtime: dict[str, Any]) -> dict[str, Any] | None:
        """Check if todos are still in context, inject reminder if missing."""
        if not runtime.get("is_plan_mode"):
            return None

        # Check if write_todos is still in the message history
        messages = getattr(state, "messages", [])
        has_todos_in_messages = any(
            m.get("tool_calls") and any(
                tc.get("function", {}).get("name") == "write_todos"
                for tc in m["tool_calls"]
            )
            for m in messages
        )

        if has_todos_in_messages or self._todos_injected:
            return None

        if not self._todos:
            return None

        # Inject reminder with current todo state
        todo_text = "\n".join(
            f"- [{'x' if t.get('status') == 'completed' else ' '}] {t.get('content', '')}"
            for t in self._todos
        )
        reminder = (
            f"<system_reminder>\n"
            f"Your todo list from earlier was truncated. Current state:\n"
            f"{todo_text}\n"
            f"Continue working on pending items.\n"
            f"</system_reminder>"
        )

        state.messages.append({"role": "user", "content": reminder})
        self._todos_injected = True
        return None

    def set_todos(self, todos: list[dict[str, str]]) -> None:
        """Update the cached todo list (called by write_todos tool)."""
        self._todos = todos
        self._todos_injected = False

    def get_todos(self) -> list[dict[str, str]]:
        return list(self._todos)

    def __repr__(self) -> str:
        return f"TodoMiddleware(todos={len(self._todos)})"
