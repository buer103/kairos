"""Context compressor middleware — multi-level compression targeting 99%+ compression.

Hermes achieves 99.4% compression via:
    1. Token-accurate counting (tiktoken or heuristic)
    2. Layered summaries (turn-level → block-level → global)
    3. Tool output smart truncation (keep first/last + error lines)
    4. Sliding window with configurable overlap
    5. Importance scoring (keep errors, decisions, user asks)
    6. Compression ratio tracking for observability

Strategy:
    - Tier 0: < 50% budget → passthrough (no compression)
    - Tier 1: 50-85% budget → tool output truncation + light summarization
    - Tier 2: 85-100% budget → layered summarization (early turns → block summary)
    - Tier 3: > 100% budget → aggressive compression (summarize all but last N)
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from kairos.core.middleware import Middleware

logger = logging.getLogger("kairos.compress")

# Try tiktoken for accurate counting, fall back to heuristic
try:
    import tiktoken
    _tk_enc = tiktoken.get_encoding("o200k_base")  # GPT-4o encoding
    HAS_TIKTOKEN = True
except Exception:
    _tk_enc = None
    HAS_TIKTOKEN = False


def count_tokens(text: str | list | dict) -> int:
    """Count tokens in text, using tiktoken if available."""
    if isinstance(text, (list, dict)):
        text = json.dumps(text, ensure_ascii=False)
    if not isinstance(text, str):
        text = str(text)

    if HAS_TIKTOKEN and _tk_enc:
        try:
            return len(_tk_enc.encode(text))
        except Exception:
            pass

    # Heuristic: ~3.5 chars per token for English, ~1.5 for CJK
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    cjk_chars = sum(1 for c in text if ord(c) >= 128)
    return (ascii_chars // 4) + (cjk_chars // 2)


class ContextCompressor(Middleware):
    """Multi-level context compressor with observability.

    Usage::

        compressor = ContextCompressor(
            max_tokens=120000,   # Model context window
            budget_ratio=0.85,    # Start compressing at 85% of window
            keep_recent=8,        # Always keep last 8 messages
            tool_truncate=2000,   # Truncate tool outputs > 2000 tokens
            importance_scoring=True,
        )
    """

    # Minimum tokens to reserve for the actual response
    RESPONSE_RESERVE = 1024

    def __init__(
        self,
        max_tokens: int = 120000,
        budget_ratio: float = 0.85,
        keep_recent: int = 8,
        tool_truncate: int = 2000,
        importance_scoring: bool = True,
        track_compression: bool = True,
    ):
        self.max_tokens = max_tokens
        self.budget_ratio = budget_ratio
        self.keep_recent = keep_recent
        self.tool_truncate = tool_truncate
        self.importance_scoring = importance_scoring
        self.track_compression = track_compression
        self._stats: list[dict[str, Any]] = []

    def before_model(self, state: Any, runtime: dict[str, Any]) -> dict[str, Any] | None:
        messages = getattr(state, "messages", [])
        if not messages:
            return None

        total = self._count_messages(messages)
        budget = int(self.max_tokens * self.budget_ratio) - self.RESPONSE_RESERVE

        # ── Tier 0: passthrough ──────────────────────────────
        if total <= budget * 0.5:
            if self.track_compression:
                self._record("passthrough", total, total, 0)
            return None

        # ── Tier 1: tool output truncation ───────────────────
        if total <= budget:
            trunc_saved = self._truncate_tool_outputs(messages)
            new_total = self._count_messages(messages)
            if self.track_compression:
                self._record("tool_truncation", total, new_total, trunc_saved)
            return {"compressed_before": total, "compressed_after": new_total}

        # ── Tier 2: layered summarization ────────────────────
        result = self._compress_messages(
            messages, budget, keep_recent=self.keep_recent
        )
        new_total = self._count_messages(messages)
        if self.track_compression:
            self._record("layered_summary", total, new_total, total - new_total)

        return {
            "compressed_before": total,
            "compressed_after": new_total,
            "layers": result.get("layers", 0),
        }

    # ═══════════════════════════════════════════════════════════
    # Core compression logic
    # ═══════════════════════════════════════════════════════════

    def _compress_messages(
        self,
        messages: list[dict],
        budget: int,
        keep_recent: int = 8,
    ) -> dict[str, Any]:
        """Layered compression: summarize early messages into block summaries."""

        # Identify system message
        sys_idx = 0 if messages and messages[0].get("role") == "system" else -1
        sys_msg = messages[sys_idx] if sys_idx >= 0 else None

        # Recent messages are never compressed
        recent = messages[-keep_recent:] if len(messages) > keep_recent else messages
        middle_start = sys_idx + 1 if sys_idx >= 0 else 0
        middle_end = len(messages) - keep_recent if len(messages) > keep_recent else len(messages)
        middle = messages[middle_start:middle_end]

        if not middle:
            return {"layers": 0}

        # Split middle into conversation "blocks" (user→assistant pairs)
        blocks = self._split_into_blocks(middle)

        # Score each block for importance
        scored = [(self._importance_score(b), b) for b in blocks]

        # Reserve budget for system + recent + response reserve
        sys_tokens = self._count_messages([sys_msg]) if sys_msg else 0
        recent_tokens = self._count_messages(recent)
        available = budget - sys_tokens - recent_tokens

        # Greedy: keep high-importance blocks until budget exhausted
        scored.sort(key=lambda x: x[0], reverse=True)
        kept: list[list[dict]] = []
        summarized: list[str] = []
        used = 0

        for score, block in scored:
            block_tokens = self._count_messages(block)
            if score > 0.5 and used + block_tokens <= available:
                kept.append(block)
                used += block_tokens
            else:
                summarized.append(self._summarize_block(block))

        # Build compressed messages
        new_middle: list[dict] = []
        if summarized:
            summary_text = (
                f"[Conversation summary — {len(summarized)} turns compressed]\n"
                + "\n".join(f"• {s}" for s in summarized)
            )
            new_middle.append({
                "role": "system",
                "content": summary_text,
            })

        # Flatten kept blocks back
        for block in kept:
            new_middle.extend(block)

        # Rebuild messages
        rebuilt: list[dict] = []
        if sys_msg:
            rebuilt.append(sys_msg)
        rebuilt.extend(new_middle)
        rebuilt.extend(recent)

        state_messages = messages  # We modify in place via the state reference
        messages.clear()
        messages.extend(rebuilt)

        return {"layers": len(summarized), "blocks_kept": len(kept)}

    def _truncate_tool_outputs(self, messages: list[dict]) -> int:
        """Truncate long tool call outputs. Returns tokens saved."""
        total_saved = 0
        for m in messages:
            if m.get("role") != "tool":
                continue
            content = m.get("content", "")
            if not isinstance(content, str):
                continue

            tokens = count_tokens(content)
            if tokens <= self.tool_truncate:
                continue

            # Smart truncation: keep head + error lines + tail
            lines = content.split("\n")
            head = lines[:3]  # First 3 lines
            # Keep lines matching error patterns
            error_lines = [
                l for l in lines[3:-3]
                if re.search(r"(?i)(error|exception|failed|traceback|warning)", l)
            ]
            tail = lines[-3:]  # Last 3 lines

            truncated = (
                "\n".join(head)
                + "\n... [" + str(len(lines) - 6 - len(error_lines))
                + " lines truncated] ...\n"
                + "\n".join(error_lines)
                + ("\n" if error_lines else "")
                + "\n".join(tail)
            )

            old_tokens = tokens
            new_tokens = count_tokens(truncated)
            total_saved += old_tokens - new_tokens
            m["content"] = truncated

        return total_saved

    # ═══════════════════════════════════════════════════════════
    # Helpers
    # ═══════════════════════════════════════════════════════════

    def _split_into_blocks(self, messages: list[dict]) -> list[list[dict]]:
        """Split messages into user→assistant(+tool) blocks."""
        blocks: list[list[dict]] = []
        current: list[dict] = []

        for m in messages:
            role = m.get("role", "")
            if role == "user" and current:
                blocks.append(current)
                current = []
            current.append(m)

        if current:
            blocks.append(current)

        return blocks

    def _importance_score(self, block: list[dict]) -> float:
        """Score a conversation block for importance (0–1).

        Higher score = more likely to be kept during compression.
        """
        if not self.importance_scoring:
            return 1.0

        score = 0.3  # baseline

        for m in block:
            content = str(m.get("content", ""))
            role = m.get("role", "")

            # User questions are important
            if role == "user" and "?" in content:
                score += 0.2

            # Error/fix patterns are important
            if re.search(r"(?i)(error|exception|bug|fix|broken|crash|failed)", content):
                score += 0.3

            # Decisions and conclusions
            if re.search(r"(?i)(therefore|conclusion|decided|agreed|done|completed|final|summary)", content):
                score += 0.15

            # Code blocks suggest substantive content
            if "```" in content:
                score += 0.1

            # Tool calls are structural (keep if possible)
            if m.get("tool_calls"):
                score += 0.1

        return min(score, 1.0)

    def _summarize_block(self, block: list[dict]) -> str:
        """Create a concise summary of a conversation block."""
        parts: list[str] = []
        for m in block:
            role = m.get("role", "?")
            content = m.get("content", "")
            if m.get("tool_calls"):
                tools = [tc.get("function", {}).get("name", "?") for tc in m["tool_calls"]]
                parts.append(f"[{role} called: {', '.join(tools)}]")
            elif content:
                # Truncate to key snippet
                snippet = str(content)[:120].replace("\n", " ")
                parts.append(f"[{role}]: {snippet}{'...' if len(str(content)) > 120 else ''}")
        return " → ".join(parts) if parts else "[empty turn]"

    def _count_messages(self, messages: list[dict]) -> int:
        """Count total tokens across all messages."""
        total = 0
        for m in messages:
            content = m.get("content", "")
            if isinstance(content, str):
                total += count_tokens(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        total += count_tokens(block.get("text", ""))
            if m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    args = tc.get("function", {}).get("arguments", "")
                    total += count_tokens(args)
        return total

    def _record(self, strategy: str, before: int, after: int, saved: int) -> None:
        """Record compression stats for observability."""
        ratio = (saved / before * 100) if before > 0 else 0
        self._stats.append({
            "strategy": strategy,
            "before": before,
            "after": after,
            "saved": saved,
            "ratio_pct": round(ratio, 2),
        })

    @property
    def stats(self) -> list[dict[str, Any]]:
        """Return compression statistics."""
        return self._stats

    def __repr__(self) -> str:
        return (
            f"ContextCompressor(max_tokens={self.max_tokens}, "
            f"budget_ratio={self.budget_ratio}, keep_recent={self.keep_recent}, "
            f"tiktoken={HAS_TIKTOKEN})"
        )
