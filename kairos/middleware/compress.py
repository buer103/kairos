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
from kairos.middleware.trajectory_compressor import TrajectoryCompressor, CompressionStats
from kairos.middleware.importance_scorer import (
    ImportanceScorer,
    RetentionPolicy,
    count_msg_tokens,
)

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
        # ── New trajectory compressor + importance scorer options ──
        use_trajectory_compressor: bool = False,
        use_importance_scorer: bool = False,
        importance_scorer_policy: RetentionPolicy | None = None,
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

        # ── Skill rescue config ────────────────────────────────
        self._skill_tool_names: set[str] = {
            "skill_view",
            "skills_list",
        }
        self._skill_dir_pattern: str = "/skills/"   # Path marker for skill file reads
        self._max_rescued_bundles: int = 5          # Max skill bundles to preserve

        # ── New: trajectory compressor + importance scorer ──────
        self.use_trajectory_compressor = use_trajectory_compressor
        self.use_importance_scorer = use_importance_scorer
        self._trajectory_compressor: TrajectoryCompressor | None = None
        self._importance_scorer: ImportanceScorer | None = None

        if use_trajectory_compressor:
            self._trajectory_compressor = TrajectoryCompressor(
                keep_recent=keep_recent,
                max_summary_tokens=max(200, budget_ratio * max_tokens // 20),
                summarize_model=summarize_model if llm_summarize else None,
                track_stats=track_compression,
            )

        if use_importance_scorer:
            self._importance_scorer = ImportanceScorer(
                policy=importance_scorer_policy or RetentionPolicy()
            )

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

        # ── NEW PATH: trajectory compressor + importance scorer ──
        if self.use_trajectory_compressor and self._trajectory_compressor:
            return self._compress_via_trajectory(messages, total, budget, runtime)

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
    # Skill Rescue (DeerFlow-compatible)
    # ═══════════════════════════════════════════════════════════

    def _rescue_skill_bundles(
        self,
        middle: list[dict],
    ) -> tuple[list[list[dict]], list[dict]]:
        """Identify and extract skill-related message bundles from middle section.

        A skill bundle is: AIMessage with skill_view/skills_list tool_call +
        the resulting ToolMessage. These must be preserved so the Agent
        doesn't lose skill context after compression.

        Returns:
            (rescued_bundles, remaining_messages) —
            rescued_bundles are guaranteed-safe blocks,
            remaining_messages are everything else.
        """
        rescued: list[list[dict]] = []
        rescued_tool_ids: set[str] = set()

        if not middle:
            return rescued, list(middle)

        # Pass 1: find AI messages calling skill tools
        skill_call_indices: list[int] = []
        for i, m in enumerate(middle):
            if m.get("role") != "assistant":
                continue
            tool_calls = m.get("tool_calls", [])
            for tc in tool_calls:
                fn_name = tc.get("function", {}).get("name", "")
                if fn_name in self._skill_tool_names:
                    skill_call_indices.append(i)
                    rescued_tool_ids.add(tc.get("id", ""))
                    break

        if not skill_call_indices:
            return rescued, list(middle)

        # Pass 2: extract skill bundles (AI call + tool response)
        remaining: list[dict] = []
        i = 0
        while i < len(middle):
            if i in skill_call_indices and len(rescued) < self._max_rescued_bundles:
                bundle = [middle[i]]
                i += 1
                # Collect following tool messages with rescued tool_call_ids
                while i < len(middle) and middle[i].get("role") == "tool":
                    tid = middle[i].get("tool_call_id", "")
                    if tid in rescued_tool_ids:
                        bundle.append(middle[i])
                        i += 1
                    else:
                        break
                rescued.append(bundle)
            else:
                remaining.append(middle[i])
                i += 1

        if rescued:
            logger.debug(
                "Skill rescue: preserved %d bundles (%d messages rescued from compression)",
                len(rescued), sum(len(b) for b in rescued),
            )

        return rescued, remaining

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

        # Skill Rescue: protect skill bundles before compression
        rescued_bundles, middle = self._rescue_skill_bundles(middle)

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

        # Skill Rescue: guaranteed-keep bundles (deduct from budget)
        for bundle in rescued_bundles:
            bundle_tokens = self._count_messages(bundle)
            kept_blocks.append(bundle)
            used += bundle_tokens
            available = max(0, available - bundle_tokens)
        rescued_count = len(rescued_bundles)

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
            "rescued_bundles": rescued_count,
        }

    # ═══════════════════════════════════════════════════════════
    # Trajectory-based compression (new, tier 2 + 3)
    # ═══════════════════════════════════════════════════════════

    def _compress_via_trajectory(
        self,
        messages: list[dict],
        total: int,
        budget: int,
        runtime: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Compress using ImportanceScorer + TrajectoryCompressor.

        Tier 2 (100-150% budget): heuristic trajectory compression.
        Tier 3 (> 150% budget): LLM-powered trajectory compression.
        """
        tc = self._trajectory_compressor
        if tc is None:
            return None

        # ── Tier detection ────────────────────────────────────
        if total <= budget:
            # We're between 50% and 100% — do tool truncation + mild compression.
            saved = self._truncate_tool_outputs(messages)
            new_total = self._count_messages(messages)
            if self.track_compression:
                self._record("tool_truncation", total, new_total, saved, runtime)
            return {"compressed_before": total, "compressed_after": new_total}

        tier2 = total <= budget * 1.5
        tier_label = "tier2_trajectory" if tier2 else "tier3_trajectory_llm"

        # ── Optional: ImportanceScorer pre-filter ─────────────
        importance_stats: dict[str, Any] = {}
        if self.use_importance_scorer and self._importance_scorer:
            # Allocate half the budget to important messages, rest for compression.
            selection_budget = budget // 2
            before_filter = len(messages)
            selected = self._importance_scorer.select_messages(
                messages, selection_budget, token_counter=count_msg_tokens,
            )
            # Replace messages in place so TrajectoryCompressor sees the selection.
            messages.clear()
            messages.extend(selected)
            importance_stats = dict(self._importance_scorer.last_stats)
            importance_stats["messages_before_filter"] = before_filter

        # ── Trajectory compression ────────────────────────────
        if tier2:
            # Tier 2: heuristic summarization.
            tc._summarize_model = None  # force heuristic
        else:
            # Tier 3: use LLM for summarization if available.
            tc._summarize_model = self._summarize_model if self.llm_summarize else None

        before = sum(count_msg_tokens(m) for m in messages)
        tc.compress(messages)
        after = sum(count_msg_tokens(m) for m in messages)
        tc_stats = tc.stats.to_dict()

        if self.track_compression:
            self._record(
                tier_label,
                total,
                after,
                max(0, total - after),
                runtime,
                trajectory_stats=tc_stats,
                importance_stats=importance_stats,
            )

        result: dict[str, Any] = {
            "compressed_before": total,
            "compressed_after": after,
            "tier": tier_label,
            "trajectory_stats": tc_stats,
        }
        if importance_stats:
            result["importance_stats"] = importance_stats

        return result

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

            # Skill rescue: max importance for skill tool calls
            if m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    fn = tc.get("function", {}).get("name", "")
                    if fn in self._skill_tool_names:
                        score = 1.0  # Never compress skill bundles
                        break

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
                runtime: dict[str, Any],
                trajectory_stats: dict | None = None,
                importance_stats: dict | None = None) -> None:
        """Record compression stats with session context."""
        ratio = (saved / before * 100) if before > 0 else 0
        compression_ratio = (after / before) if before > 0 else 1.0
        entry: dict[str, Any] = {
            "strategy": strategy,
            "before": before,
            "after": after,
            "saved": saved,
            "ratio_pct": round(ratio, 2),
            "compression_ratio": round(compression_ratio, 4),
            "thread_id": runtime.get("thread_id", ""),
            "timestamp": __import__("time").time(),
        }
        if trajectory_stats:
            entry["trajectory"] = trajectory_stats
            entry["messages_retained"] = trajectory_stats.get("messages_compressed", 0)
        if importance_stats:
            entry["importance"] = importance_stats
            entry["avg_importance"] = importance_stats.get("avg_importance_kept", 0.0)
            entry["messages_retained"] = importance_stats.get("messages_retained", entry.get("messages_retained", 0))
        self._stats.append(entry)

    @property
    def stats(self) -> list[dict[str, Any]]:
        return self._stats

    def __repr__(self) -> str:
        return (
            f"ContextCompressor(max_tokens={self.max_tokens}, "
            f"budget_ratio={self.budget_ratio}, keep_recent={self.keep_recent}, "
            f"tiktoken={HAS_TIKTOKEN}, llm={self.llm_summarize}, "
            f"trajectory={self.use_trajectory_compressor}, "
            f"importance_scorer={self.use_importance_scorer})"
        )
