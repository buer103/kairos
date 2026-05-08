"""Context compressor middleware — summarizes when token budget is tight."""

from __future__ import annotations

from typing import Any

from kairos.core.middleware import Middleware


class ContextCompressor(Middleware):
    """Summarizes early messages when approaching token limits."""

    def __init__(self, max_tokens: int = 8000, threshold: float = 0.85):
        self.max_tokens = max_tokens
        self.threshold = threshold

    def before_model(self, state: Any, runtime: dict[str, Any]) -> dict[str, Any] | None:
        estimated = sum(len(str(m.get("content", ""))) // 4 for m in state.messages)
        if estimated > self.max_tokens * self.threshold:
            # Future: summarize early messages, keep recent ones intact
            pass
        return None

    def __repr__(self) -> str:
        return f"ContextCompressor(max_tokens={self.max_tokens}, threshold={self.threshold})"
