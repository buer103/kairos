"""Dangling tool call middleware — fixes broken tool calls in message history.

When a user interrupts or a tool times out, the message history may contain
AIMessage tool_calls without matching ToolMessage results, causing LLM errors.
This middleware scans for and patches such dangling calls before model invocation.

DeerFlow layer 4 — must run before any model call.
"""

from __future__ import annotations

from typing import Any

from kairos.core.middleware import Middleware


class DanglingToolCallMiddleware(Middleware):
    """Fixes dangling tool calls by injecting synthetic ToolMessage placeholders.

    Hook: wrap_model_call — needs to insert at specific positions in message list,
    not just append at the end (which before_model would do).

    Scenarios that create dangling calls:
      1. User interrupts mid-tool-execution
      2. Tool execution times out
      3. Server restart during tool execution
    """

    def wrap_model_call(self, messages: list[dict], handler, **kwargs) -> Any:
        patched = self._patch(messages)
        return handler(patched, **kwargs)

    def _patch(self, messages: list[dict]) -> list[dict]:
        """Scan for dangling tool calls and inject placeholder ToolMessages."""
        # Collect all tool_call_ids from assistant messages
        expected_ids: set[str] = set()
        for m in messages:
            if m.get("role") == "assistant" and m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    expected_ids.add(tc["id"])

        # Collect all tool_call_ids from tool messages
        resolved_ids: set[str] = set()
        for m in messages:
            if m.get("role") == "tool" and m.get("tool_call_id"):
                resolved_ids.add(m["tool_call_id"])

        dangling_ids = expected_ids - resolved_ids
        if not dangling_ids:
            return messages

        # Build patched message list, inserting synthetic ToolMessages
        # right after the assistant message that generated the dangling call
        patched = []
        for m in messages:
            patched.append(m)

            if m.get("role") == "assistant" and m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    if tc["id"] in dangling_ids:
                        tool_name = tc.get("function", {}).get("name", "unknown")
                        patched.append({
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "name": tool_name,
                            "content": "[Tool call was interrupted and did not return a result.]",
                        })

        return patched

    def __repr__(self) -> str:
        return "DanglingToolCallMiddleware()"
