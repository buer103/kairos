"""Middleware pipeline — 6 hook types, composable layers.

Built-in middleware implementations live in kairos/middleware/.
This module provides only the base class and pipeline orchestrator.
"""

from __future__ import annotations

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
                    if hasattr(state, k):
                        setattr(state, k, v)

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
            prev = handler
            handler = lambda msgs, **kw: mw.wrap_model_call(msgs, prev, **kw)
        return handler(messages, **kwargs)

    def wrap_tool_call(self, tool_name, args, handler, **kwargs):
        for mw in reversed(self._layers):
            prev = handler
            handler = lambda n, a, **kw: mw.wrap_tool_call(n, a, prev, **kw)
        return handler(tool_name, args, **kwargs)
