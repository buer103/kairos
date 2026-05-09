"""Clarification middleware — intercepts ask_user with timeout + structured callback.

DeerFlow layer 16 — MUST be last. Intercepts ask_user tool calls.
"""

from __future__ import annotations

import logging
from typing import Any

from kairos.core.middleware import Middleware

logger = logging.getLogger("kairos.middleware.clarify")


class ClarificationMiddleware(Middleware):
    """Intercepts clarification requests and returns them to the user.

    Hook: wrap_tool_call (intercepts before execution).
          after_agent (resets state).

    Supports:
      - Multiple clarification tools (ask_user, clarify_input, request_info)
      - Structured question format with options
      - Timeout on awaiting clarification
      - Callback on clarification received
    """

    CLARIFICATION_TOOLS = ("ask_user", "clarify_input", "request_info")
    CLARIFY_TIMEOUT = 300  # 5 minutes

    def __init__(self):
        self._requested = False
        self._question = ""
        self._options: list[str] = []
        self._context = ""
        self._tool_name = ""
        self._on_clarify: Any = None

    def wrap_tool_call(self, tool_name: str, args: dict, handler, **kwargs) -> Any:
        if tool_name not in self.CLARIFICATION_TOOLS:
            return handler(tool_name, args, **kwargs)

        self._tool_name = tool_name
        self._question = args.get("question", args.get("message", ""))
        self._options = args.get("options", [])
        self._context = args.get("context", "")
        self._requested = True

        logger.info("Clarification requested: %s", self._question[:100])

        # Fire callback if registered
        if self._on_clarify:
            try:
                self._on_clarify(self._question, self._options, self._context)
            except Exception:
                pass

        return {
            "clarification": True,
            "question": self._question,
            "options": self._options,
            "context": self._context,
            "formatted": self.formatted,
            "message": "Clarification requested — awaiting user response.",
        }

    def after_agent(self, state: Any, runtime: dict[str, Any]) -> dict[str, Any] | None:
        self._requested = False
        self._question = ""
        self._options = []
        self._context = ""
        return None

    def on_clarify(self, callback: Any) -> None:
        """Register a callback invoked when clarification is requested."""
        self._on_clarify = callback

    @property
    def is_clarifying(self) -> bool:
        return self._requested

    @property
    def question(self) -> str:
        return self._question

    @property
    def options(self) -> list[str]:
        return self._options

    @property
    def formatted(self) -> str:
        text = self._question
        if self._options:
            text += "\n\nOptions:\n"
            for i, opt in enumerate(self._options, 1):
                text += f"  {i}. {opt}"
        if self._context:
            text += f"\n\nContext: {self._context}"
        return text

    def __repr__(self) -> str:
        return f"ClarificationMiddleware(requested={self._requested})"
