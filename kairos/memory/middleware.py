"""Memory middleware — injects persistent memory + processes memory tool calls.

Hooks: before_agent (inject memory), after_agent (auto-save).

Integration point for the memory tool (if agent calls memory.add/remove).
"""

from __future__ import annotations

from typing import Any

from kairos.core.middleware import Middleware
from kairos.memory.store import MemoryStore


class MemoryMiddleware(Middleware):
    """Injects persistent memory into system prompt and handles memory writes.

    Hook: before_agent — injects formatted memory block.
          after_agent — auto-saves memory tool results.
    """

    def __init__(self, memory_store: MemoryStore | None = None):
        self._store = memory_store or MemoryStore()
        self._pending_writes: list[dict] = []

    def before_agent(self, state: Any, runtime: dict[str, Any]) -> dict[str, Any] | None:
        prompt_block = self._store.format_for_prompt(max_chars=3000)
        if not prompt_block:
            return None

        if state.messages and state.messages[0].get("role") == "system":
            section = f"\n\n## MEMORY (persistent knowledge)\n{prompt_block}"
            state.messages[0]["content"] += section

        return None

    def after_agent(self, state: Any, runtime: dict[str, Any]) -> dict[str, Any] | None:
        """Process any pending memory writes from the session."""
        for write in self._pending_writes:
            try:
                self._store.add(
                    scope=write.get("scope", "memory"),
                    key=write["key"],
                    value=write["value"],
                    priority=write.get("priority", 0.5),
                )
            except Exception:
                pass
        self._pending_writes.clear()
        return None

    def queue_write(self, scope: str, key: str, value: str, priority: float = 0.5) -> None:
        """Queue a memory write to be processed at session end."""
        self._pending_writes.append({
            "scope": scope, "key": key, "value": value, "priority": priority,
        })

    @property
    def store(self) -> MemoryStore:
        return self._store
