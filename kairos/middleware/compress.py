"""Context compressor middleware — summarizes early messages when near token limits."""

from __future__ import annotations

from typing import Any

from kairos.core.middleware import Middleware


class ContextCompressor(Middleware):
    """Summarizes early messages when approaching token budget limits.

    Hook: before_model — checks token estimate before each LLM call.
    Strategy: keep system prompt + last N messages intact, summarize middle.

    Estimation: ~4 chars ≈ 1 token (rough heuristic, no tokenizer dependency).
    """

    def __init__(self, max_tokens: int = 8000, threshold: float = 0.85, keep_recent: int = 6):
        self.max_tokens = max_tokens
        self.threshold = threshold
        self.keep_recent = keep_recent

    def before_model(self, state: Any, runtime: dict[str, Any]) -> dict[str, Any] | None:
        messages = getattr(state, "messages", [])
        if not messages:
            return None

        total = self._estimate_tokens(messages)
        limit = int(self.max_tokens * self.threshold)

        if total <= limit:
            return None

        # Keep: system message (if present) + last N messages
        system_msg = messages[0] if messages and messages[0].get("role") == "system" else None
        recent = messages[-self.keep_recent:] if len(messages) > self.keep_recent else messages
        middle = messages[1:-self.keep_recent] if system_msg else messages[:-self.keep_recent]

        if not middle:
            return None

        # Summarize middle messages into one compressed message
        summary_parts = []
        for m in middle:
            role = m.get("role", "unknown")
            content = m.get("content", "")
            if isinstance(content, str) and content:
                content = content[:80] + ("..." if len(content) > 80 else "")
            summary_parts.append(f"[{role}]: {content}")

        summary = " ".join(summary_parts)
        if len(summary) > 500:
            summary = summary[:500] + "..."

        compacted_msg = {
            "role": "system",
            "content": f"[Compressed context — {len(middle)} messages summarized]\n{summary}",
        }

        # Rebuild: system + compacted + recent
        rebuilt = []
        if system_msg:
            rebuilt.append(system_msg)
        rebuilt.append(compacted_msg)
        rebuilt.extend(recent)

        state.messages = rebuilt
        runtime["compressed"] = True
        return {"compressed_before": total, "compressed_after": self._estimate_tokens(rebuilt)}

    @staticmethod
    def _estimate_tokens(messages: list[dict]) -> int:
        """Rough token estimate from message content lengths."""
        total = 0
        for m in messages:
            content = m.get("content", "")
            total += len(str(content)) // 4 if content else 0
            # Tool calls add overhead
            if m.get("tool_calls"):
                total += sum(
                    len(str(tc.get("function", {}).get("arguments", ""))) // 4
                    for tc in m["tool_calls"]
                )
        return total

    def __repr__(self) -> str:
        return (
            f"ContextCompressor(max_tokens={self.max_tokens}, "
            f"threshold={self.threshold}, keep_recent={self.keep_recent})"
        )
