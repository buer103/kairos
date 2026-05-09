"""Title middleware — auto-generates a conversational title for the session.

After the first complete user-assistant exchange, generates a concise title.
Fallbacks to first 50 chars of user message if generation is unavailable.

DeerFlow layer 7 — runs after first complete interaction.
"""

from __future__ import annotations

from typing import Any

from kairos.core.middleware import Middleware


class TitleMiddleware(Middleware):
    """Generates a session title from the first conversation exchange.

    Hook: after_agent — checks if this is the first exchange and generates a title.

    Title generation uses a simple heuristic (first user message). In production,
    this would call a lightweight LLM for better titles.
    """

    MAX_TITLE_LENGTH = 80

    def __init__(self):
        self._title: str = ""
        self._title_generated = False

    def before_agent(self, state: Any, runtime: dict[str, Any]) -> dict[str, Any] | None:
        """Reset for new conversation."""
        self._title = ""
        self._title_generated = False
        return None

    def after_agent(self, state: Any, runtime: dict[str, Any]) -> dict[str, Any] | None:
        """Generate title after first complete exchange."""
        if self._title_generated:
            return None

        messages = getattr(state, "messages", [])
        # Find the first user message
        user_msg = ""
        for m in messages:
            if m.get("role") == "user" and m.get("content"):
                user_msg = m["content"]
                break

        if not user_msg:
            return None

        # Simple heuristic: first line or first 50 chars
        title = user_msg.split("\n")[0].strip()
        if len(title) > self.MAX_TITLE_LENGTH:
            title = title[:self.MAX_TITLE_LENGTH - 3] + "..."

        self._title = title
        self._title_generated = True

        runtime["title"] = title
        return None

    @property
    def title(self) -> str:
        return self._title

    def __repr__(self) -> str:
        return f"TitleMiddleware(title={self._title!r})"
