"""Logging middleware — structured JSON observability for every pipeline hook.

Logs agent lifecycle, model calls, and tool executions with latency stats.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from kairos.core.middleware import Middleware

logger = logging.getLogger("kairos.middleware.observability")


class LoggingMiddleware(Middleware):
    """Structured observability across the full agent lifecycle.

    Hooks: before_agent, after_agent, before_model, after_model, wrap_tool_call.
    Tracks: latency (p50/p95), tool counts, error rates.
    """

    def __init__(self):
        self._session_start = 0.0
        self._tool_count = 0
        self._error_count = 0
        self._tool_latencies: list[float] = []
        self._model_count = 0

    def before_agent(self, state: Any, runtime: dict[str, Any]) -> dict[str, Any] | None:
        self._session_start = time.time()
        self._tool_count = 0
        self._error_count = 0
        self._tool_latencies = []
        self._model_count = 0

        self._log("session_start", {
            "thread_id": runtime.get("thread_id", ""),
            "turn": runtime.get("turn", 0),
        })
        return None

    def after_agent(self, state: Any, runtime: dict[str, Any]) -> dict[str, Any] | None:
        elapsed = time.time() - self._session_start
        p50, p95 = self._percentiles(self._tool_latencies)

        self._log("session_end", {
            "thread_id": runtime.get("thread_id", ""),
            "duration_s": round(elapsed, 2),
            "tool_count": self._tool_count,
            "error_count": self._error_count,
            "model_calls": self._model_count,
            "tool_latency_p50_ms": round(p50, 1),
            "tool_latency_p95_ms": round(p95, 1),
            "error_rate": round(self._error_count / max(self._tool_count, 1), 3),
        })
        return None

    def before_model(self, state: Any, runtime: dict[str, Any]) -> dict[str, Any] | None:
        runtime["_model_start"] = time.time()
        return None

    def after_model(self, state: Any, runtime: dict[str, Any]) -> dict[str, Any] | None:
        start = runtime.get("_model_start", time.time())
        elapsed = (time.time() - start) * 1000
        self._model_count += 1

        if elapsed > 30000:
            logger.warning("Slow model call: %.1fs", elapsed / 1000)
        return None

    def wrap_tool_call(self, tool_name: str, args: dict, handler, **kwargs) -> Any:
        start = time.time()
        error = None
        try:
            result = handler(tool_name, args, **kwargs)
        except Exception as e:
            error = str(e)
            result = {"error": error}
            raise
        finally:
            elapsed = (time.time() - start) * 1000
            self._tool_count += 1
            if error:
                self._error_count += 1
            self._tool_latencies.append(elapsed)

        return result

    def _log(self, event: str, data: dict) -> None:
        data["event"] = event
        data["timestamp"] = time.time()
        logger.info(json.dumps(data, ensure_ascii=False, default=str))

    @staticmethod
    def _percentiles(values: list[float]) -> tuple[float, float]:
        if not values:
            return 0.0, 0.0
        s = sorted(values)
        n = len(s)
        p50 = s[n // 2]
        p95 = s[min(int(n * 0.95), n - 1)]
        return p50, p95

    def __repr__(self) -> str:
        return f"LoggingMiddleware(tools={self._tool_count}, models={self._model_count})"
