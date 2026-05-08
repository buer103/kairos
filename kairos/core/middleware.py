"""Middleware pipeline — 6 hook types, composable layers."""

from __future__ import annotations

import time
from abc import ABC
from typing import Any


class Middleware(ABC):
    """Base class for middleware. Override any hook you need."""

    # --- Agent lifecycle ---
    def before_agent(self, state: Any, runtime: dict[str, Any]) -> dict[str, Any] | None:
        """Called once, before the agent starts processing."""
        return None

    def after_agent(self, state: Any, runtime: dict[str, Any]) -> dict[str, Any] | None:
        """Called once, after the agent finishes."""
        return None

    # --- Model lifecycle ---
    def before_model(self, state: Any, runtime: dict[str, Any]) -> dict[str, Any] | None:
        """Called before every LLM call."""
        return None

    def after_model(self, state: Any, runtime: dict[str, Any]) -> dict[str, Any] | None:
        """Called after every LLM call."""
        return None

    def wrap_model_call(self, messages: list[dict], handler, **kwargs) -> Any:
        """Wrap the LLM call. Can modify messages before sending."""
        return handler(messages, **kwargs)

    # --- Tool lifecycle ---
    def wrap_tool_call(self, tool_name: str, args: dict, handler, **kwargs) -> Any:
        """Wrap tool execution. Can intercept, record, or modify."""
        return handler(tool_name, args, **kwargs)


class MiddlewarePipeline:
    """Ordered chain of middleware layers."""

    def __init__(self, layers: list[Middleware] | None = None):
        self._layers: list[Middleware] = layers or []

    def add(self, middleware: Middleware) -> "MiddlewarePipeline":
        self._layers.append(middleware)
        return self

    # --- Run hooks in order ---
    def before_agent(self, state, runtime):
        for mw in self._layers:
            result = mw.before_agent(state, runtime)
            if result:
                for k, v in result.items():
                    setattr(state, k, v) if hasattr(state, k) else None

    def after_agent(self, state, runtime):
        for mw in self._layers:
            mw.after_agent(state, runtime)

    def before_model(self, state, runtime):
        for mw in self._layers:
            mw.before_model(state, runtime)

    def after_model(self, state, runtime):
        for mw in self._layers:
            mw.after_model(state, runtime)

    def wrap_model_call(self, messages, handler, **kwargs):
        for mw in self._layers:
            prev_handler = handler
            handler = lambda msgs, **kw: mw.wrap_model_call(msgs, prev_handler, **kw)
        return handler(messages, **kwargs)

    def wrap_tool_call(self, tool_name, args, handler, **kwargs):
        for mw in reversed(self._layers):
            prev_handler = handler
            handler = lambda name, a, **kw: mw.wrap_tool_call(name, a, prev_handler, **kw)
        return handler(tool_name, args, **kwargs)


# --- Built-in middleware implementations ---

class EvidenceTracker(Middleware):
    """Records every tool invocation as a Step in the evidence chain."""

    def wrap_tool_call(self, tool_name, args, handler, **kwargs):
        state = kwargs.get("state")
        start = time.time()
        result = handler(tool_name, args, **kwargs)
        elapsed = (time.time() - start) * 1000

        if state and state.case:
            step = state.case.add_step(tool_name, args)
            state.case.complete_step(step, result, elapsed)

        return result


class ConfidenceScorer(Middleware):
    """Evaluates output confidence and attaches an evidence summary."""

    def after_agent(self, state, runtime):
        if not state.case or not state.case.steps:
            return
        # Simple heuristic: more steps with results = higher confidence
        completed = [s for s in state.case.steps if s.result is not None]
        if not completed:
            return
        # Confidence based on evidence chain completeness
        state.case.confidence = min(0.5 + (len(completed) / max(len(state.case.steps), 1)) * 0.5, 0.99)


class ContextCompressor(Middleware):
    """Summarizes early messages when approaching token limits."""

    def __init__(self, max_tokens: int = 8000, threshold: float = 0.85):
        self.max_tokens = max_tokens
        self.threshold = threshold

    def before_model(self, state, runtime):
        # Placeholder: estimate tokens and compress if needed
        estimated = sum(len(str(m.get("content", ""))) // 4 for m in state.messages)
        if estimated > self.max_tokens * self.threshold:
            # Future: summarize early messages, keep recent ones
            pass

    def __repr__(self):
        return f"ContextCompressor(max_tokens={self.max_tokens}, threshold={self.threshold})"
