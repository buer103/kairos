"""Loop Detection Middleware — DeerFlow-compatible infinite loop prevention.

Detects when the agent repeats the same tool call sequence and terminates early.

DeerFlow equivalent: LoopDetectionMiddleware
"""

from __future__ import annotations

import logging
from typing import Any

from kairos.core.middleware import Middleware

logger = logging.getLogger("kairos.middleware.loop_detection")


class LoopDetectionMiddleware(Middleware):
    """Detect and break infinite loop patterns in agent tool calls.

    Two detection strategies:
      1. Exact repetition: same tool+args combo N times consecutively
      2. Sequence repetition: same sequence of tool calls repeated

    Hooks: after_model — checks after each model response.

    Config:
        max_repeats: int — consecutive identical calls before breaking (default: 3)
        sequence_window: int — window size for sequence repetition (default: 8)
        break_message: str — message to inject when loop detected
    """

    def __init__(
        self,
        max_repeats: int = 3,
        sequence_window: int = 8,
        break_message: str | None = None,
    ):
        self._max_repeats = max_repeats
        self._sequence_window = sequence_window
        self._break_message = break_message or (
            "Loop detected — the same action has been repeated multiple times. "
            "Please try a different approach or ask for clarification."
        )
        self._history: list[str] = []  # Tool call fingerprints
        self._loop_broken = False

    def after_model(self, state: Any, runtime: dict[str, Any]) -> dict[str, Any] | None:
        messages = getattr(state, "messages", [])
        if not messages:
            return None

        last = messages[-1]
        if last.get("role") != "assistant":
            return None

        tool_calls = last.get("tool_calls", [])
        if not tool_calls:
            # No tool calls — reset pattern tracking
            self._history.clear()
            return None

        # Build fingerprint for this turn
        fingerprints = [self._fingerprint(tc) for tc in tool_calls]
        self._history.extend(fingerprints)

        # Check exact repetition
        if self._detect_exact_repeat():
            return self._break_loop(state)
        if self._detect_sequence_repeat():
            return self._break_loop(state)

    def _fingerprint(self, tool_call: dict) -> str:
        """Create a stable fingerprint from a tool call."""
        fn = tool_call.get("function", {})
        name = fn.get("name", "unknown")
        args = fn.get("arguments", "{}")
        # Normalize: sort args if JSON
        try:
            import json
            parsed = json.loads(args) if isinstance(args, str) else args
            if isinstance(parsed, dict):
                normalized = json.dumps(parsed, sort_keys=True)
            else:
                normalized = str(args)
        except Exception:
            normalized = str(args)
        return f"{name}:{normalized}"

    def _detect_exact_repeat(self) -> bool:
        """Check if the last N fingerprints are all the same."""
        if len(self._history) < self._max_repeats:
            return False
        recent = self._history[-self._max_repeats:]
        return len(set(recent)) == 1

    def _detect_sequence_repeat(self) -> bool:
        """Check if a sequence of tool calls is repeating."""
        if len(self._history) < self._sequence_window:
            return False
        half = self._sequence_window // 2
        first_half = tuple(self._history[-self._sequence_window:-half])
        second_half = tuple(self._history[-half:])
        if len(first_half) < 2 or len(second_half) < 2:
            return False
        return first_half == second_half

    def _break_loop(self, state: Any) -> dict[str, Any] | None:
        """Inject break message and remove tool calls."""
        messages = getattr(state, "messages", [])
        last = messages[-1] if messages else {}
        tool_count = len(last.get("tool_calls", []))

        # Remove tool calls to prevent execution
        last["tool_calls"] = []
        if not last.get("content"):
            last["content"] = self._break_message
        else:
            last["content"] = f"{self._break_message}\n\n{last['content']}"

        self._loop_broken = True
        logger.warning(
            "LoopDetection: broken after %d tool calls (%d in last turn)",
            len(self._history), tool_count,
        )

        # Reset history after breaking
        self._history.clear()

        return {"loop_broken": True, "tool_calls_removed": tool_count}

    @property
    def loop_broken(self) -> bool:
        return self._loop_broken

    def reset(self) -> None:
        """Reset pattern history."""
        self._history.clear()
        self._loop_broken = False

    def __repr__(self) -> str:
        return (
            f"LoopDetectionMiddleware(max_repeats={self._max_repeats}, "
            f"window={self._sequence_window})"
        )
