"""Context compressor — multi-tier token budget management with LLM summarization.

Target: 99%+ compression via layered strategy (Hermes benchmark).

Tiers:
  Tier 0: < 50% budget → passthrough
  Tier 1: 50-100% budget → tool output truncation
  Tier 2: 100-150% budget → heuristic block summary
  Tier 3: > 150% budget → LLM-based layered compression

Key fixes over v0.8.0:
  - Safe budget calculation (no negative `available`)
  - LLM-based summarization with heuristic fallback
  - BeforeCompressionHook for downstream middleware
  - Proper message ID tracking for deletion
  - Session-tagged stats
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable

from kairos.core.middleware import Middleware

logger = logging.getLogger("kairos.middleware.compress")

# ── Token counting ─────────────────────────────────────────────

try:
    import tiktoken
    _tk_enc = tiktoken.get_encoding("o200k_base")
    HAS_TIKTOKEN = True
except Exception:
    _tk_enc = None
    HAS_TIKTOKEN = False


def count_tokens(text: str | list | dict) -> int:
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

    # Heuristic: ~4 chars per token for ASCII, ~2 for CJK
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    cjk_chars = sum(1 for c in text if ord(c) >= 128)
    return (ascii_chars // 4) + (cjk_chars // 2)


# ── Hook protocol ───────────────────────────────────────────────

class BeforeCompressionHook:
    """Hook called before messages are compressed away.

    Downstream middleware (Todo, Evidence, Memory) can register to save
    state before their messages are summarized out of context.
    """

    def __call__(self, messages_to_compress: list[dict], runtime: dict) -> None:
        pass


# ── Compressor ──────────────────────────────────────────────────

class ContextCompressor(Middleware):
    """Multi-tier context compressor with LLM-based summarization.

    Usage:
        compressor = ContextCompressor(
            max_tokens=120000,
            budget_ratio=0.85,
            keep_recent=8,
            llm_summarize=True,       # Use LLM for summaries (requires model)
        )
    """

    RESPONSE_RESERVE = 1024  # Minimum tokens reserved for the model's response

    def __init__(
        self,
        max_tokens: int = 120000,
        budget_ratio: float = 0.85,
        keep_recent: int = 8,
        tool_truncate: int = 2000,
        importance_scoring: bool = True,
        llm_summarize: bool = False,
        summarize_model: Any = None,  # ModelProvider for LLM summaries
        track_compression: bool = True,
    ):
        self.max_tokens = max_tokens
        self.budget_ratio = budget_ratio
        self.keep_recent = keep_recent
        self.tool_truncate = tool_truncate
        self.importance_scoring = importance_scoring
        self.llm_summarize = llm_summarize
        self._summarize_model = summarize_model
        self.track_compression = track_compression
        self._stats: list[dict] = []
        self._hooks: list[BeforeCompressionHook] = []

    # ── Hook registration ──────────────────────────────────────

    def add_hook(self, hook: BeforeCompressionHook) -> None:
        """Register a hook called before messages are compressed away."""
        self._hooks.append(hook)

    # ── before_model — main entry ──────────────────────────────

    def before_model(self, state: Any, runtime: dict[str, Any]) -> dict[str, Any] | None:
        messages = getattr(state, "messages", [])
        if not messages:
            return None

        total = self._count_messages(messages)
        budget = max(0, int(self.max_tokens * self.budget_ratio) - self.RESPONSE_RESERVE)

        # ── Tier 0: passthrough ──────────────────────────────
        if total <= budget * 0.5:
            if self.track_compression:
                self._record("passthrough", total, total, 0, runtime)
            return None

        # ── Tier 1: tool output truncation ───────────────────
        if total <= budget:
            saved = self._truncate_tool_outputs(messages)
            new_total = self._count_messages(messages)
            if self.track_compression:
                self._record("tool_truncation", total, new_total, saved, runtime)
            return {"compressed_before": total, "compressed_after": new_total}

        # ── Tier 2 & 3: block compression ────────────────────
        result = self._compress_messages(messages, budget, runtime)
        new_total = self._count_messages(messages)
        if self.track_compression:
            self._record(
                result.get("tier", "compressed"),
                total, new_total, max(0, total - new_total),
                runtime,
            )

        return {
            "compressed_before": total,
            "compressed_after": new_total,
            "tier": result.get("tier"),
            "blocks_kept": result.get("blocks_kept", 0),
            "blocks_summarized": result.get("blocks_summarized", 0),
        }

    # ═══════════════════════════════════════════════════════════
    # Core compression
    # ═══════════════════════════════════════════════════════════

    def _compress_messages(
        self,
        messages: list[dict],
        budget: int,
        runtime: dict[str, Any],
    ) -> dict[str, Any]:
        """Layered compression with LLM summarization option."""
        sys_idx = 0 if messages and messages[0].get("role") == "system" else -1
        sys_msg = messages[sys_idx] if sys_idx >= 0 else None

        recent = messages[-self.keep_recent:] if len(messages) > self.keep_recent else messages
        middle_start = sys_idx + 1 if sys_idx >= 0 else 0
        middle_end = len(messages) - self.keep_recent if len(messages) > self.keep_recent else len(messages)
        middle = messages[middle_start:middle_end]

        if not middle:
            return {"tier": "passthrough", "blocks_kept": 0, "blocks_summarized": 0}

        # Split into conversation blocks
        blocks = self._split_into_blocks(middle)

        # Reserve budget
        sys_tokens = self._count_messages([sys_msg]) if sys_msg else 0
        recent_tokens = self._count_messages(recent)
        available = max(0, budget - sys_tokens - recent_tokens)

        # Notify hooks before compression
        for hook in self._hooks:
            try:
                hook(middle, runtime)
            except Exception:
                pass

        # Score and sort blocks
        if self.importance_scoring:
            scored = [(self._importance_score(b), i, b) for i, b in enumerate(blocks)]
            scored.sort(key=lambda x: (-x[0], x[1]))  # high score first, stable
        else:
            scored = [(1.0, i, b) for i, b in enumerate(blocks)]

        # Greedy allocation
        kept_blocks: list[list[dict]] = []
        summarized_blocks: list[list[dict]] = []
        used = 0

        for score, _idx, block in scored:
            block_tokens = self._count_messages(block)
            if score > 0.4 and used + block_tokens <= available:
                kept_blocks.append(block)
                used += block_tokens
            else:
                summarized_blocks.append(block)

        # Build compressed message list
        rebuilt: list[dict] = []
        if sys_msg:
            rebuilt.append(sys_msg)

        if summarized_blocks:
            summary = self._build_summary(summarized_blocks)
            rebuilt.append({
                "role": "system",
                "content": summary,
                "name": "compression_summary",
            })

        for block in kept_blocks:
            rebuilt.extend(block)

        rebuilt.extend(recent)

        # Replace in place
        messages.clear()
        messages.extend(rebuilt)

        tier = "tier3_llm" if self.llm_summarize and summarized_blocks else "tier2_heuristic"
        return {
            "tier": tier,
            "blocks_kept": len(kept_blocks),
            "blocks_summarized": len(summarized_blocks),
        }

    def _build_summary(self, blocks: list[list[dict]]) -> str:
        """Build a summary of compressed blocks.

        Uses LLM if configured, otherwise falls back to heuristic extraction.
        """
        if not blocks:
            return ""

        if self.llm_summarize and self._summarize_model:
            return self._llm_summarize(blocks)

        return self._heuristic_summarize(blocks)

    def _llm_summarize(self, blocks: list[list[dict]]) -> str:
        """Use the LLM to produce a concise summary of compressed blocks."""
        conversation = []
        for block in blocks:
            for msg in block:
                role = msg.get("role", "?")
                content = msg.get("content", "")
                if msg.get("tool_calls"):
                    tools = [tc.get("function", {}).get("name", "?") for tc in msg.get("tool_calls", [])]
                    conversation.append(f"[{role} called: {', '.join(tools)}]")
                elif content:
                    conversation.append(f"[{role}]: {str(content)[:300]}")
                elif msg.get("role") == "tool":
                    conversation.append(f"[tool result]: {str(msg.get('content', ''))[:200]}")

        prompt = (
            "Summarize the following conversation excerpt in 2-4 sentences, "
            "preserving key decisions, code changes, errors, and conclusions. "
            "Be concise and factual.\n\n"
            + "\n".join(conversation)
        )

        try:
            response = self._summarize_model.chat(
                [{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0,
            )
            return (
                f"[Conversation summary — {len(blocks)} turns compressed]\n"
                + response.choices[0].message.content
            )
        except Exception as e:
            logger.debug("LLM summarization failed, falling back to heuristic: %s", e)
            return self._heuristic_summarize(blocks)

    def _heuristic_summarize(self, blocks: list[list[dict]]) -> str:
        """Heuristic summary by extracting key snippets from each block."""
        lines = [f"[Conversation summary — {len(blocks)} turns compressed]"]

        for i, block in enumerate(blocks):
            parts = []
            for m in block:
                role = m.get("role", "?")
                content = str(m.get("content", ""))
                if m.get("tool_calls"):
                    tools = [tc.get("function", {}).get("name", "?") for tc in m["tool_calls"]]
                    parts.append(f"[{role} called: {', '.join(tools)}]")
                elif content:
                    snippet = content[:120].replace("\n", " ")
                    suffix = "..." if len(content) > 120 else ""
                    parts.append(f"[{role}]: {snippet}{suffix}")
            if parts:
                lines.append(f"• {' → '.join(parts)}")

        return "\n".join(lines)

    # ═══════════════════════════════════════════════════════════
    # Tool output truncation
    # ═══════════════════════════════════════════════════════════

    def _truncate_tool_outputs(self, messages: list[dict]) -> int:
        """Smart truncation: keep head + error lines + tail."""
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

            lines = content.split("\n")
            if len(lines) <= 10:
                continue

            head = lines[:3]
            error_lines = [
                l for l in lines[3:-3]
                if re.search(r"(?i)(error|exception|failed|traceback|warning)", l)
            ]
            tail = lines[-3:]

            truncated = (
                "\n".join(head)
                + f"\n... [{len(lines) - 6 - len(error_lines)} lines truncated] ...\n"
                + "\n".join(error_lines)
                + ("\n" if error_lines else "")
                + "\n".join(tail)
            )

            total_saved += tokens - count_tokens(truncated)
            m["content"] = truncated

        return total_saved

    # ═══════════════════════════════════════════════════════════
    # Helpers
    # ═══════════════════════════════════════════════════════════

    def _split_into_blocks(self, messages: list[dict]) -> list[list[dict]]:
        """Split flat message list into user→assistant(+tool) conversation blocks."""
        blocks = []
        current = []
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
        """Score a block's importance (0-1). Higher → keep."""
        if not self.importance_scoring:
            return 1.0

        score = 0.3
        for m in block:
            content = str(m.get("content", ""))
            role = m.get("role", "")

            if role == "user" and "?" in content:
                score += 0.2
            if re.search(r"(?i)(error|exception|bug|fix|broken|crash|failed)", content):
                score += 0.3
            if re.search(r"(?i)(therefore|conclusion|decided|agreed|done|completed|final|summary)", content):
                score += 0.15
            if "```" in content:
                score += 0.1
            if m.get("tool_calls"):
                score += 0.1
            if role == "user":
                score += 0.05  # Slight preference for keeping user messages

        return min(score, 1.0)

    def _count_messages(self, messages: list[dict]) -> int:
        """Count tokens across all messages including tool calls."""
        total = 0
        for m in messages:
            # Role overhead (~4 tokens)
            total += 4

            content = m.get("content", "")
            if isinstance(content, str):
                total += count_tokens(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        total += count_tokens(block.get("text", ""))

            if m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    fn = tc.get("function", {})
                    if fn:
                        total += count_tokens(fn.get("name", ""))
                        total += count_tokens(fn.get("arguments", ""))

            if m.get("tool_call_id"):
                total += count_tokens(m["tool_call_id"])
                total += count_tokens(m.get("name", ""))

        return total

    def _record(self, strategy: str, before: int, after: int, saved: int,
                runtime: dict[str, Any]) -> None:
        """Record compression stats with session context."""
        ratio = (saved / before * 100) if before > 0 else 0
        self._stats.append({
            "strategy": strategy,
            "before": before,
            "after": after,
            "saved": saved,
            "ratio_pct": round(ratio, 2),
            "thread_id": runtime.get("thread_id", ""),
            "timestamp": __import__("time").time(),
        })

    @property
    def stats(self) -> list[dict[str, Any]]:
        return self._stats

    def __repr__(self) -> str:
        return (
            f"ContextCompressor(max_tokens={self.max_tokens}, "
            f"budget_ratio={self.budget_ratio}, keep_recent={self.keep_recent}, "
            f"tiktoken={HAS_TIKTOKEN}, llm={self.llm_summarize})"
        )
