"""Clarification middleware — intercepts ask_user tool calls to pause execution.

When the agent doesn't have enough info, it calls ask_user(clarification_question).
This middleware intercepts that call and returns it as the final output instead
of executing it as a normal tool.

DeerFlow layer 11 — MUST be the last middleware, uses interrupt/goto-end pattern.
"""

from __future__ import annotations

import json
from typing import Any

from kairos.core.middleware import Middleware


class ClarificationMiddleware(Middleware):
    """Intercepts clarification requests and returns them to the user.

    Hook: wrap_tool_call — intercepts before tool execution.

    When the agent calls ask_user(), instead of executing it as a tool,
    we return the clarification question as the agent's response so the
    user can answer it in the next turn.
    """

    CLARIFICATION_TOOL = "ask_user"

    def __init__(self):
        self._clarification_requested = False
        self._clarification_text = ""

    def wrap_tool_call(self, tool_name: str, args: dict, handler, **kwargs) -> Any:
        """Intercept ask_user tool calls."""
        if tool_name != self.CLARIFICATION_TOOL:
            return handler(tool_name, args, **kwargs)

        # Intercept: don't execute, instead return as a special response
        question = args.get("question", args.get("message", ""))
        options = args.get("options", [])
        context = args.get("context", "")

        formatted = question
        if options:
            formatted += "\n\nOptions:\n"
            for i, opt in enumerate(options, 1):
                formatted += f"  {i}. {opt}"
        if context:
            formatted += f"\n\nContext: {context}"

        self._clarification_requested = True
        self._clarification_text = formatted

        # Return a tool result that signals clarification is needed
        return {
            "clarification": True,
            "question": question,
            "options": options,
            "formatted": formatted,
            "message": "Clarification requested — awaiting user response.",
        }

    def after_agent(self, state: Any, runtime: dict[str, Any]) -> dict[str, Any] | None:
        """Reset state after agent finishes."""
        self._clarification_requested = False
        self._clarification_text = ""
        return None

    @property
    def is_clarifying(self) -> bool:
        return self._clarification_requested

    @property
    def question(self) -> str:
        return self._clarification_text

    def __repr__(self) -> str:
        return "ClarificationMiddleware()"
