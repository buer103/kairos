"""Hook system — publish/subscribe lifecycle hooks for Kairos.

Hermes-aligned: 24 named hook points at key lifecycle moments.
Plugins and middleware register callbacks that fire at specific points.

Usage::

    from kairos.hooks import HookPoint, get_hook_registry

    registry = get_hook_registry()

    @registry.on(HookPoint.BEFORE_TOOL)
    def audit_tool_call(tool_name, args, **ctx):
        print(f"Tool call: {tool_name}({args})")

    # Later in agent loop:
    registry.emit(HookPoint.BEFORE_TOOL, tool_name="read_file", args={"path": "x.py"})
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger("kairos.hooks")


class HookPoint(str, Enum):
    """Named lifecycle hook points — 24 total."""

    # ── Agent lifecycle ──
    AGENT_START = "agent:start"         # Agent.run() called
    AGENT_END = "agent:end"             # Agent.run() returning
    AGENT_INTERRUPT = "agent:interrupt"  # SIGINT received

    # ── Model lifecycle ──
    BEFORE_MODEL = "model:before"       # Before LLM call
    AFTER_MODEL = "model:after"         # After LLM response
    MODEL_ERROR = "model:error"         # LLM call failed

    # ── Tool lifecycle ──
    BEFORE_TOOL = "tool:before"         # Before tool execution
    AFTER_TOOL = "tool:after"           # After tool succeeds
    TOOL_ERROR = "tool:error"           # Tool execution failed
    TOOL_RETRY = "tool:retry"           # Grace call retry attempt

    # ── Message lifecycle ──
    BEFORE_MESSAGE = "message:before"   # Before message sent to model
    AFTER_MESSAGE = "message:after"     # After message received from model

    # ── Session ──
    SESSION_SAVE = "session:save"       # Session persisted
    SESSION_LOAD = "session:load"       # Session loaded

    # ── Middleware chain ──
    MIDDLEWARE_CHAIN_START = "middleware:chain_start"  # Pipeline begins
    MIDDLEWARE_CHAIN_END = "middleware:chain_end"      # Pipeline ends

    # ── Compression ──
    BEFORE_COMPRESSION = "compress:before"  # Before context compression
    AFTER_COMPRESSION = "compress:after"    # After compression applied

    # ── Memory ──
    MEMORY_SAVE = "memory:save"         # Fact saved to memory
    MEMORY_LOAD = "memory:load"         # Memory loaded for injection

    # ── Gateway ──
    GATEWAY_MESSAGE_RECEIVED = "gateway:message_received"  # Incoming message
    GATEWAY_RESPONSE_SENT = "gateway:response_sent"        # Outgoing response

    # ── Skills ──
    SKILL_LOADED = "skill:loaded"       # Skill document loaded
    SKILL_CREATED = "skill:created"     # New skill created


# ── Registry ────────────────────────────────────────────────────


class HookRegistry:
    """Thread-safe publish/subscribe registry for hook callbacks.

    Supports:
      - Multiple subscribers per hook point
      - Async and sync callbacks
      - Error isolation (one failing callback doesn't block others)
      - Priority ordering (lower = earlier execution)
    """

    def __init__(self):
        self._hooks: dict[HookPoint, list[tuple[int, Callable]]] = defaultdict(list)
        self._lock = threading.Lock()

    def on(
        self, hook: HookPoint, priority: int = 100
    ) -> Callable[[Callable], Callable]:
        """Decorator: register a callback for a hook point.

        Args:
            hook: The lifecycle hook point to subscribe to.
            priority: Lower numbers execute first (default: 100).

        Usage::

            registry = get_hook_registry()

            @registry.on(HookPoint.BEFORE_TOOL, priority=50)
            def my_auditor(tool_name, args, **ctx):
                ...
        """

        def decorator(fn: Callable) -> Callable:
            self.register(hook, fn, priority=priority)
            return fn

        return decorator

    def register(
        self, hook: HookPoint, callback: Callable, priority: int = 100
    ) -> None:
        """Register a callback for a hook point.

        Thread-safe. Can be called at any time.
        """
        with self._lock:
            self._hooks[hook].append((priority, callback))
            # Keep sorted by priority
            self._hooks[hook].sort(key=lambda x: x[0])

    def unregister(self, hook: HookPoint, callback: Callable) -> bool:
        """Remove a callback from a hook point. Returns True if found."""
        with self._lock:
            before = len(self._hooks[hook])
            self._hooks[hook] = [
                (p, cb) for p, cb in self._hooks[hook] if cb is not callback
            ]
            return len(self._hooks[hook]) < before

    def emit(self, hook: HookPoint, **context: Any) -> list[Any]:
        """Fire all callbacks registered for a hook point.

        Each callback receives **context as keyword arguments.
        Returns list of callback return values (excluding None).

        Errors in individual callbacks are caught and logged —
        one failing callback does not prevent others from running.
        """
        results: list[Any] = []
        with self._lock:
            callbacks = self._hooks.get(hook, [])[:]

        for _priority, callback in callbacks:
            try:
                result = callback(**context)
                if result is not None:
                    results.append(result)
            except Exception as e:
                logger.warning(
                    "Hook %s callback %s failed: %s",
                    hook.value, getattr(callback, "__name__", callback), e,
                )

        return results

    def emit_async(self, hook: HookPoint, **context: Any) -> list[Any]:
        """Async-compatible emit — delegates to sync emit."""
        return self.emit(hook, **context)

    def listeners(self, hook: HookPoint) -> int:
        """Number of registered callbacks for a hook point."""
        with self._lock:
            return len(self._hooks.get(hook, []))

    def clear(self) -> None:
        """Remove all registered hooks."""
        with self._lock:
            self._hooks.clear()

    def list_hooks(self) -> dict[str, int]:
        """Return {hook_name: listener_count} for all hooks with listeners."""
        with self._lock:
            return {
                h.value: len(cbs)
                for h, cbs in self._hooks.items()
                if cbs
            }


# ── Global singleton ────────────────────────────────────────────

_registry: HookRegistry | None = None
_registry_lock = threading.Lock()


def get_hook_registry() -> HookRegistry:
    """Get or create the global HookRegistry singleton."""
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                _registry = HookRegistry()
    return _registry


def reset_hook_registry() -> None:
    """Reset the global registry (for testing)."""
    global _registry
    with _registry_lock:
        _registry = None
