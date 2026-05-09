"""Title middleware — auto-generates session titles from first exchange.

DeerFlow layer 14. Uses LLM-assisted generation with heuristic fallback.
"""

from __future__ import annotations

import logging
from typing import Any

from kairos.core.middleware import Middleware

logger = logging.getLogger("kairos.middleware.title")

MAX_TITLE_LENGTH = 80


class TitleMiddleware(Middleware):
    """Generates a concise session title from the first user-assistant exchange.

    Hook: after_agent — checks if first exchange, generates title.
    """

    def __init__(self, llm_generate: bool = False, title_model: Any = None):
        self._title = ""
        self._generated = False
        self._llm_generate = llm_generate
        self._model = title_model

    def before_agent(self, state: Any, runtime: dict[str, Any]) -> dict[str, Any] | None:
        self._title = ""
        self._generated = False
        return None

    def after_agent(self, state: Any, runtime: dict[str, Any]) -> dict[str, Any] | None:
        if self._generated:
            return None

        messages = getattr(state, "messages", [])

        # Extract first user message AND first assistant response
        user_msg = ""
        assistant_msg = ""
        for m in messages:
            if m.get("role") == "user" and m.get("content") and not user_msg:
                user_msg = m["content"]
            if m.get("role") == "assistant" and m.get("content") and user_msg:
                assistant_msg = m["content"]
                break

        if not user_msg:
            return None

        # LLM-assisted generation
        if self._llm_generate and self._model and assistant_msg:
            title = self._llm_title(user_msg, assistant_msg)
        else:
            title = self._heuristic_title(user_msg)

        self._title = title[:MAX_TITLE_LENGTH]
        self._generated = True
        runtime["title"] = self._title

        state.metadata = state.metadata or {}
        state.metadata["session_title"] = self._title

        logger.debug("Generated title: %s", self._title)
        return None

    def _heuristic_title(self, text: str) -> str:
        """Extract a title from the first line or first sentence."""
        # Take first meaningful line (skip code blocks, empty lines)
        for line in text.split("\n"):
            line = line.strip()
            if not line or line.startswith("```") or line.startswith("#"):
                continue
            if len(line) > 10:
                if len(line) > MAX_TITLE_LENGTH:
                    return line[:MAX_TITLE_LENGTH - 3] + "..."
                return line
        return text[:MAX_TITLE_LENGTH]

    def _llm_title(self, user_msg: str, assistant_msg: str) -> str:
        """Generate a concise title using an LLM."""
        prompt = (
            "Generate a concise title (max 80 chars) for this conversation:\n"
            f"User: {user_msg[:200]}\n"
            f"Assistant: {assistant_msg[:200]}\n"
            "Title:"
        )
        try:
            resp = self._model.chat(
                [{"role": "user", "content": prompt}],
                max_tokens=20, temperature=0,
            )
            return resp.choices[0].message.content.strip().strip('"')
        except Exception as e:
            logger.debug("LLM title generation failed: %s", e)
            return self._heuristic_title(user_msg)

    @property
    def title(self) -> str:
        return self._title

    def __repr__(self) -> str:
        return f"TitleMiddleware(title={self._title!r})"
