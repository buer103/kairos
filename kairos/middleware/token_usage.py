"""Token Usage Middleware — DeerFlow-compatible per-message token attribution.

Injects token/cost metadata into each AI message's additional_kwargs.
Categorizes each turn: final_answer, thinking, tool_batch, todo_update,
subagent_dispatch, search.

DeerFlow equivalent: TokenUsageMiddleware (294 lines)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from kairos.core.middleware import Middleware

logger = logging.getLogger("kairos.middleware.token_usage")


@dataclass
class TokenUsage:
    """Per-turn token usage statistics."""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    turn_index: int = 0
    kind: str = "unknown"
    tool_names: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    timestamp: float = 0.0


class TokenUsageMiddleware(Middleware):
    """Track token usage and inject attribution metadata into AI messages.

    Hook: after_model — inspects the last AI message and stamps it with:
      - token_usage_attribution in additional_kwargs
      - kind (final_answer / thinking / tool_batch / search / subagent)
      - actions array (tool names called)
      - cost estimate

    Config:
        price_per_1k_input: float — base input price per 1k tokens
        price_per_1k_output: float — base output price per 1k tokens
        track_history: bool — accumulate per-turn stats
    """

    def __init__(
        self,
        price_per_1k_input: float = 0.001,   # ~GPT-3.5 level
        price_per_1k_output: float = 0.002,
        track_history: bool = True,
    ):
        self._input_price = price_per_1k_input
        self._output_price = price_per_1k_output
        self._track_history = track_history
        self._history: list[TokenUsage] = []
        self._total_input = 0
        self._total_output = 0
        self._total_cost = 0.0
        self._turn = 0

    def after_model(self, state: Any, runtime: dict[str, Any]) -> dict[str, Any] | None:
        messages = getattr(state, "messages", [])
        if not messages:
            return None

        last = messages[-1]
        if last.get("role") != "assistant":
            return None

        self._turn += 1
        t0 = time.time()

        # Extract token counts from message if available
        usage = last.get("usage", {}) or runtime.get("last_usage", {})
        input_tokens = usage.get("input_tokens", usage.get("prompt_tokens", 0))
        output_tokens = usage.get("output_tokens", usage.get("completion_tokens", 0))
        total_tokens = usage.get("total_tokens", input_tokens + output_tokens)

        # Classify kind
        tool_calls = last.get("tool_calls", [])
        content = last.get("content", "")

        if tool_calls:
            tool_names = [
                tc.get("function", {}).get("name", "unknown")
                for tc in tool_calls
            ]
            # Detect subagent dispatch
            if any("delegate" in n or "task" in n for n in tool_names):
                kind = "subagent_dispatch"
            elif any("search" in n for n in tool_names):
                kind = "search"
            elif any("todo" in n for n in tool_names):
                kind = "todo_update"
            else:
                kind = "tool_batch"
        elif content:
            kind = "final_answer"
        else:
            kind = "thinking"

        # Calculate cost
        cost = (
            (input_tokens / 1000) * self._input_price
            + (output_tokens / 1000) * self._output_price
        )

        # Build attribution metadata
        attribution = {
            "kind": kind,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "cost_usd": round(cost, 6),
            "turn_index": self._turn,
            "tool_names": tool_names if tool_calls else [],
            "latency_ms": round((time.time() - t0) * 1000, 1),
            "timestamp": time.time(),
        }

        # Inject into message
        akw = last.get("additional_kwargs", {})
        if not isinstance(akw, dict):
            akw = {}
        akw["token_usage_attribution"] = attribution
        last["additional_kwargs"] = akw

        # Track history
        self._total_input += input_tokens
        self._total_output += output_tokens
        self._total_cost += cost

        if self._track_history:
            self._history.append(TokenUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                cost_usd=cost,
                turn_index=self._turn,
                kind=kind,
                tool_names=tool_names if tool_calls else [],
            ))

        return None

    # ── Properties ──────────────────────────────────────────

    @property
    def history(self) -> list[TokenUsage]:
        return list(self._history)

    @property
    def total_input(self) -> int:
        return self._total_input

    @property
    def total_output(self) -> int:
        return self._total_output

    @property
    def total_cost(self) -> float:
        return self._total_cost

    @property
    def turn_count(self) -> int:
        return self._turn

    def __repr__(self) -> str:
        return (
            f"TokenUsageMiddleware(turns={self._turn}, "
            f"tokens={self._total_input + self._total_output}, "
            f"cost=${self._total_cost:.4f})"
        )
