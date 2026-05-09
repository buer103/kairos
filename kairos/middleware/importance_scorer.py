"""Importance scorer — per-message retention priority for context compression.

Scores each message 0.0–1.0 on how valuable it is to keep in the limited
context window.  Provides a greedy selector that packs the highest-scoring
messages within a token budget.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger("kairos.middleware.importance_scorer")

# ── Token counting (shared) ──────────────────────────────────────

try:
    import tiktoken

    _tk_enc = tiktoken.get_encoding("o200k_base")
    HAS_TIKTOKEN = True
except Exception:
    _tk_enc = None
    HAS_TIKTOKEN = False


def _count_tokens(text: str | list | dict) -> int:
    """Count tokens using tiktoken, falling back to heuristic."""
    if isinstance(text, (list, dict)):
        text = json.dumps(text, ensure_ascii=False)
    if not isinstance(text, str):
        text = str(text)

    if HAS_TIKTOKEN and _tk_enc:
        try:
            return len(_tk_enc.encode(text))
        except Exception:
            pass

    ascii_chars = sum(1 for c in text if ord(c) < 128)
    cjk_chars = sum(1 for c in text if ord(c) >= 128)
    return (ascii_chars // 4) + (cjk_chars // 2)


def count_msg_tokens(msg: dict) -> int:
    """Count tokens for a single message including role overhead."""
    total = 4  # role overhead
    content = msg.get("content", "")
    if isinstance(content, str):
        total += _count_tokens(content)
    elif isinstance(content, list):
        for block in content:
            if isinstance(block, dict):
                total += _count_tokens(block.get("text", ""))
    if msg.get("tool_calls"):
        for tc in msg["tool_calls"]:
            fn = tc.get("function", {})
            if fn:
                total += _count_tokens(fn.get("name", ""))
                total += _count_tokens(fn.get("arguments", ""))
    if msg.get("tool_call_id"):
        total += _count_tokens(msg["tool_call_id"])
        total += _count_tokens(msg.get("name", ""))
    return total


# ── Retention policy ─────────────────────────────────────────────


@dataclass
class RetentionPolicy:
    """Configurable scoring weights for ImportanceScorer.

    All weights are additive and clamped to [0.0, 1.0] after summation.
    """

    base_user: float = 0.8
    base_system: float = 1.0
    base_tool_error: float = 0.9
    base_tool_success: float = 0.3
    base_assistant_toolcall: float = 0.4
    base_assistant_response: float = 0.7
    base_unknown: float = 0.3

    # Boosts / penalties
    boost_keyword: float = 0.2       # error, fix, bug, critical, etc.
    boost_recent: float = 0.1        # near conversation end
    penalty_short: float = -0.2      # very short messages (< 10 chars)
    short_threshold: int = 10        # chars below which penalty applies

    # Keyword patterns that trigger a boost.
    keyword_patterns: list[str] = field(default_factory=lambda: [
        "error", "exception", "fix", "bug", "critical", "crash",
        "broken", "failed", "urgent", "important", "security",
        "vulnerability", "decision", "conclusion",
    ])


# ── Importance Scorer ────────────────────────────────────────────


class ImportanceScorer:
    """Score messages for retention priority and select a token-budgeted subset.

    Usage::

        scorer = ImportanceScorer()
        selected = scorer.select_messages(messages, max_tokens=50000)
        print(f"Kept {len(selected)} / {len(messages)} messages")
        print(f"Average importance: {scorer.last_stats['avg_importance']:.2f}")
    """

    def __init__(self, policy: RetentionPolicy | None = None) -> None:
        self.policy = policy or RetentionPolicy()
        self.last_stats: dict[str, Any] = {}

    # ── Public API ───────────────────────────────────────────────

    def score_message(self, msg: dict, position: int = -1,
                      total_messages: int = 1) -> float:
        """Score a single message 0.0–1.0 for retention priority.

        Args:
            msg: The message dict (role, content, tool_calls, etc.).
            position: 0-based index of this message in the conversation.
            total_messages: Total number of messages (for recency boost).

        Returns:
            Float in [0.0, 1.0]; higher = more important to keep.
        """
        p = self.policy
        role = msg.get("role", "").lower()
        content = str(msg.get("content", ""))
        has_tool_calls = bool(msg.get("tool_calls"))

        # ── Base score by role ──────────────────────────────────
        if role == "system":
            score = p.base_system
        elif role == "user":
            score = p.base_user
        elif role == "tool":
            # Tool results with errors are more important.
            if self._has_error(content):
                score = p.base_tool_error
            else:
                score = p.base_tool_success
        elif role == "assistant":
            if has_tool_calls:
                score = p.base_assistant_toolcall
            else:
                score = p.base_assistant_response
        else:
            score = p.base_unknown

        # ── Keyword boost ───────────────────────────────────────
        if self._matches_keywords(content):
            score += p.boost_keyword

        # ── Recency boost ───────────────────────────────────────
        if position >= 0 and total_messages > 0:
            # Last 25% of messages get a boost.
            recency_fraction = position / max(total_messages - 1, 1)
            if recency_fraction >= 0.75:
                score += p.boost_recent

        # ── Short-message penalty ───────────────────────────────
        if len(content) < p.short_threshold and role != "system":
            score += p.penalty_short

        return max(0.0, min(1.0, score))

    def score_messages(self, messages: list[dict]) -> list[tuple[dict, float]]:
        """Score all messages, returning (msg, score) pairs."""
        total = len(messages)
        return [
            (msg, self.score_message(msg, position=i, total_messages=total))
            for i, msg in enumerate(messages)
        ]

    def select_messages(
        self,
        messages: list[dict],
        max_tokens: int,
        token_counter: Callable[[dict], int] | None = None,
    ) -> list[dict]:
        """Select the highest-scoring messages that fit within *max_tokens*.

        Rules:
          - System message is always kept (never dropped).
          - Last 3 messages are always kept (recency guard).
          - Remaining messages are selected greedily by descending score.

        Args:
            messages: Full message list.
            max_tokens: Token budget (inclusive).
            token_counter: Optional custom token counter (dict → int).
                           Defaults to :func:`count_msg_tokens`.

        Returns:
            New list of selected messages in original conversation order.
        """
        if not messages:
            return []

        if token_counter is None:
            token_counter = count_msg_tokens

        total = len(messages)
        scored: list[tuple[int, dict, float]] = []

        # Score every message, tracking its original index.
        for i, msg in enumerate(messages):
            s = self.score_message(msg, position=i, total_messages=total)
            scored.append((i, msg, s))

        # ── Identify mandatory messages ─────────────────────────
        mandatory_indices: set[int] = set()

        # System message (first role=="system" without a name prefix).
        for i, msg in enumerate(messages):
            if msg.get("role") == "system" and not msg.get("name"):
                mandatory_indices.add(i)
                break

        # Last 3 messages.
        for i in range(max(0, total - 3), total):
            mandatory_indices.add(i)

        # ── Mandatory tokens consumed ───────────────────────────
        mandatory_msgs = [m for i, m in enumerate(messages) if i in mandatory_indices]
        mandatory_tokens = sum(token_counter(m) for m in mandatory_msgs)

        if mandatory_tokens > max_tokens:
            # Edge case: even mandatory messages exceed budget.
            # Return only mandatory messages (system first, then most recent).
            logger.warning(
                "Mandatory messages (%d tokens) exceed budget (%d tokens) — "
                "returning mandatory-only subset.",
                mandatory_tokens, max_tokens,
            )
            result = []
            for i in sorted(mandatory_indices):
                result.append(messages[i])
            self._record_stats(scored, mandatory_indices, result)
            return result

        available = max_tokens - mandatory_tokens

        # ── Greedy selection of optional messages ───────────────
        # Sort by descending score, then by original index for stability.
        optional = [
            (idx, msg, score) for idx, msg, score in scored
            if idx not in mandatory_indices
        ]
        optional.sort(key=lambda x: (-x[2], x[0]))

        kept_optional_indices: set[int] = set()
        used = 0

        for idx, msg, _score in optional:
            t = token_counter(msg)
            if used + t <= available:
                kept_optional_indices.add(idx)
                used += t

        # ── Reconstruct in original order ───────────────────────
        all_kept = mandatory_indices | kept_optional_indices
        result = [msg for i, msg in enumerate(messages) if i in all_kept]

        self._record_stats(scored, all_kept, result)
        return result

    # ── Helpers ─────────────────────────────────────────────────

    @staticmethod
    def _has_error(content: str) -> bool:
        """Detect error indicators in tool result content."""
        low = content.lower()
        return any(
            kw in low
            for kw in ("error", "exception", "traceback", "failed", "refused",
                       "denied", "invalid", "not found", "cannot", "unable")
        )

    def _matches_keywords(self, content: str) -> bool:
        low = content.lower()
        for pat in self.policy.keyword_patterns:
            if pat in low:
                return True
        return False

    def _record_stats(
        self,
        scored: list[tuple[int, dict, float]],
        kept_indices: set[int],
        result: list[dict],
    ) -> None:
        """Record selection statistics for inspection."""
        scores = [s for _idx, _msg, s in scored]
        kept_scores = [
            s for idx, _msg, s in scored
            if idx in kept_indices
        ]
        self.last_stats = {
            "total_messages": len(scored),
            "messages_retained": len(result),
            "avg_importance_all": round(sum(scores) / max(len(scores), 1), 4),
            "avg_importance_kept": (
                round(sum(kept_scores) / max(len(kept_scores), 1), 4)
                if kept_scores else 0.0
            ),
            "min_score_kept": round(min(kept_scores), 4) if kept_scores else 0.0,
            "max_score_dropped": round(
                max(s for idx, _msg, s in scored if idx not in kept_indices),
                4,
            ) if len(kept_indices) < len(scored) else 0.0,
        }

    # ── Inspection ───────────────────────────────────────────────

    @property
    def avg_importance(self) -> float:
        return self.last_stats.get("avg_importance_kept", 0.0)

    @property
    def messages_retained(self) -> int:
        return self.last_stats.get("messages_retained", 0)

    def __repr__(self) -> str:
        return (
            f"ImportanceScorer(base_user={self.policy.base_user}, "
            f"base_tool_error={self.policy.base_tool_error})"
        )
