"""Memory middleware — injects persistent memory into agent context."""

from __future__ import annotations

from typing import Any

from kairos.core.middleware import Middleware
from kairos.memory.store import MemoryStore


class MemoryMiddleware(Middleware):
    """Injects persistent memory into the system prompt at session start.

    Hook: before_agent — loads memories and injects into system message.
    """

    def __init__(self, memory_store: MemoryStore | None = None):
        self._store = memory_store or MemoryStore()

    def before_agent(self, state: Any, runtime: dict[str, Any]) -> dict[str, Any] | None:
        """Inject memory block into the system prompt."""
        prompt_block = self._store.format_for_prompt()
        if not prompt_block:
            return None

        if state.messages and state.messages[0].get("role") == "system":
            memory_section = f"\n\n## MEMORY (persistent knowledge from past sessions)\n{prompt_block}"
            state.messages[0]["content"] += memory_section

        return None

    @property
    def store(self) -> MemoryStore:
        return self._store
