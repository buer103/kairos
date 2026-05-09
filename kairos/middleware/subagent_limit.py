"""Subagent limit middleware — caps concurrent task() tool calls.

Even when the system prompt specifies limits, LLMs may still generate more
task() calls than allowed. This middleware is the hard enforcement layer.

DeerFlow layer 10 — runs after_model to truncate excess tool calls.
"""

from __future__ import annotations

from typing import Any

from kairos.core.middleware import Middleware


class SubagentLimitMiddleware(Middleware):
    """Truncates excess task() tool calls from model responses.

    Hook: wrap_model_call — modifies the response before tool execution.

    Args:
        max_concurrent: hard cap on concurrent sub-agent calls (default: 3)
    """

    def __init__(self, max_concurrent: int = 3):
        self.max_concurrent = max(1, min(max_concurrent, 4))

    def wrap_model_call(self, messages: list[dict], handler, **kwargs) -> Any:
        response = handler(messages, **kwargs)

        # Only intervene if the response has tool calls
        if not hasattr(response, "choices") or not response.choices:
            return response

        msg = response.choices[0].message
        if not msg.tool_calls:
            return response

        # Count task() calls
        task_indices: list[int] = []
        for i, tc in enumerate(msg.tool_calls):
            if tc.function.name == "task":
                task_indices.append(i)

        if len(task_indices) <= self.max_concurrent:
            return response

        # Drop excess task calls (keep the first N, drop the rest)
        indices_to_drop = set(task_indices[self.max_concurrent:])
        msg.tool_calls = [
            tc for i, tc in enumerate(msg.tool_calls)
            if i not in indices_to_drop
        ]

        return response

    def __repr__(self) -> str:
        return f"SubagentLimitMiddleware(max_concurrent={self.max_concurrent})"
