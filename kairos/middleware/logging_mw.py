"""Logging middleware — logs agent lifecycle events."""

from __future__ import annotations

import time
from typing import Any

from kairos.core.middleware import Middleware
from kairos.logging import log_agent_event, log_tool_call


class LoggingMiddleware(Middleware):
    """Logs agent lifecycle events for observability.

    Hooks:
      - before_agent: log session start
      - after_agent: log session end with stats
      - wrap_tool_call: log each tool invocation with timing
    """

    def __init__(self):
        self._session_start = 0.0
        self._tool_count = 0

    def before_agent(self, state: Any, runtime: dict[str, Any]) -> dict[str, Any] | None:
        self._session_start = time.time()
        self._tool_count = 0
        session_id = runtime.get("session_id", runtime.get("thread_id", ""))

        log_agent_event(
            "info",
            "Session started",
            session_id=session_id,
            turn=runtime.get("turn", 0),
        )
        return None

    def after_agent(self, state: Any, runtime: dict[str, Any]) -> dict[str, Any] | None:
        elapsed = time.time() - self._session_start
        session_id = runtime.get("session_id", runtime.get("thread_id", ""))

        log_agent_event(
            "info",
            f"Session ended: {self._tool_count} tools, {elapsed:.1f}s",
            session_id=session_id,
            tool_count=self._tool_count,
            duration_s=round(elapsed, 1),
        )
        return None

    def wrap_tool_call(self, tool_name: str, args: dict, handler, **kwargs) -> Any:
        start = time.time()
        result = handler(tool_name, args, **kwargs)
        elapsed = (time.time() - start) * 1000
        self._tool_count += 1

        state = kwargs.get("state")
        session_id = ""
        if state and hasattr(state, "metadata"):
            session_id = state.metadata.get("session_id", "")

        log_tool_call(
            tool_name,
            args,
            result,
            elapsed,
            session_id=session_id,
        )
        return result

    def __repr__(self) -> str:
        return f"LoggingMiddleware(tools={self._tool_count})"
