"""Trajectory compressor — summarise old conversation prefixes.

Compresses conversation history by keeping recent messages intact and
summarising older ones via LLM, preserving tool-call pairs as units.

Integrates with ContextCompressor for tier 2/3 compression strategies.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger("kairos.middleware.trajectory_compressor")

# ── Token counting (shared with compress.py) ─────────────────────

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


def _msg_tokens(msg: dict) -> int:
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


# ── Summary block ────────────────────────────────────────────────


@dataclass
class SummaryBlock:
    """A compressed summary of one or more original messages."""

    summary_text: str
    original_message_ids: list[int] = field(default_factory=list)
    original_message_count: int = 0
    token_count_before: int = 0
    token_count_after: int = 0
    compress_tier: str = "heuristic"

    @property
    def token_count_saved(self) -> int:
        return max(0, self.token_count_before - self.token_count_after)


# ── Compression stats ────────────────────────────────────────────


@dataclass
class CompressionStats:
    """Aggregate statistics for trajectory compression."""

    total_tokens_before: int = 0
    total_tokens_after: int = 0
    messages_compressed: int = 0
    summary_count: int = 0
    summaries: list[SummaryBlock] = field(default_factory=list)

    @property
    def total_tokens_saved(self) -> int:
        return max(0, self.total_tokens_before - self.total_tokens_after)

    @property
    def compression_ratio(self) -> float:
        if self.total_tokens_before > 0:
            return self.total_tokens_after / self.total_tokens_before
        return 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_tokens_before": self.total_tokens_before,
            "total_tokens_after": self.total_tokens_after,
            "total_tokens_saved": self.total_tokens_saved,
            "compression_ratio": round(self.compression_ratio, 4),
            "messages_compressed": self.messages_compressed,
            "summary_count": self.summary_count,
        }


# ── Trajectory message ID generator ──────────────────────────────

_id_counter: int = 0


def _next_msg_id() -> int:
    global _id_counter
    _id_counter += 1
    return _id_counter


def _assign_ids(messages: list[dict]) -> list[dict]:
    """Assign a synthetic tracking ID to every message that lacks one."""
    for m in messages:
        if "_kairos_msg_id" not in m:
            m["_kairos_msg_id"] = _next_msg_id()
    return messages


# ── Trajectory Compressor ────────────────────────────────────────


class TrajectoryCompressor:
    """Compress conversation history by summarising old messages.

    Strategy:
      - Keep the most recent N messages verbatim (keep_recent).
      - Split the older prefix into tool-call *units* (assistant tool_calls
        + corresponding tool results stay together).
      - Summarise old units with an LLM (or heuristic fallback).
      - System message is always preserved at the top.

    Usage::

        tc = TrajectoryCompressor(
            keep_recent=10,
            max_summary_tokens=500,
            summarize_model=my_model,
        )
        compressed = tc.compress(messages)
        print(tc.stats.to_dict())
    """

    # Default system prompt used when summarising via LLM.
    SUMMARIZE_SYSTEM = (
        "You are a precise conversation summariser. Your task is to compress "
        "a segment of conversation history into a concise, factual summary. "
        "Preserve: (1) key user requests, (2) tool calls and their results, "
        "(3) errors or exceptions, (4) important decisions or conclusions. "
        "Output only the summary text — no preamble, no commentary."
    )

    def __init__(
        self,
        keep_recent: int = 10,
        max_summary_tokens: int = 500,
        summarize_model: Any = None,
        track_stats: bool = True,
    ) -> None:
        self.keep_recent = keep_recent
        self.max_summary_tokens = max_summary_tokens
        self._summarize_model = summarize_model
        self.track_stats = track_stats
        self.stats = CompressionStats()
        self._last_summaries: list[SummaryBlock] = []

    # ── Public API ───────────────────────────────────────────────

    def compress(self, messages: list[dict]) -> list[dict]:
        """Compress *messages* in place and return the compressed list.

        The original list is mutated (cleared + extended) for compatibility
        with the ContextCompressor pattern, but the return value is also
        provided for standalone use.
        """
        if not messages:
            return messages

        self.stats = CompressionStats()
        self._last_summaries = []

        # Assign tracking IDs so we can report which messages were summarised.
        _assign_ids(messages)

        before_total = sum(_msg_tokens(m) for m in messages)

        # ── 1. Extract and preserve system message ──────────────
        sys_msg: dict | None = None
        sys_index: int = -1
        for i, m in enumerate(messages):
            if m.get("role") == "system" and not m.get("name"):
                sys_msg = m
                sys_index = i
                break

        # ── 2. Split into recent (keep verbatim) and older prefix ──
        if len(messages) <= self.keep_recent:
            # Nothing to compress.
            self.stats.total_tokens_before = before_total
            self.stats.total_tokens_after = before_total
            return messages

        recent = messages[-self.keep_recent:]

        # Older prefix: everything from after system to before recent.
        prefix_start = max(0, sys_index + 1) if sys_index >= 0 else 0
        prefix_end = len(messages) - self.keep_recent
        prefix = messages[prefix_start:max(prefix_start, prefix_end)]

        if not prefix:
            self.stats.total_tokens_before = before_total
            self.stats.total_tokens_after = before_total
            return messages

        # ── 3. Group prefix into tool-call units ────────────────
        units = self._group_into_units(prefix)

        # ── 4. Summarise each unit ──────────────────────────────
        summary_texts: list[str] = []
        compressed_ids: list[int] = []
        total_compressed = 0

        for unit in units:
            unit_tokens = sum(_msg_tokens(m) for m in unit)
            unit_ids = [m.get("_kairos_msg_id", -1) for m in unit]
            summary = self._summarize_unit(unit)

            summary_tokens = _count_tokens(summary)
            sb = SummaryBlock(
                summary_text=summary,
                original_message_ids=unit_ids,
                original_message_count=len(unit),
                token_count_before=unit_tokens,
                token_count_after=summary_tokens,
                compress_tier="llm" if self._summarize_model else "heuristic",
            )
            self._last_summaries.append(sb)
            summary_texts.append(summary)
            compressed_ids.extend(unit_ids)
            total_compressed += len(unit)

        # ── 5. Rebuild the message list ─────────────────────────
        rebuilt: list[dict] = []

        if sys_msg is not None:
            rebuilt.append(sys_msg)

        # Insert summaries as system messages with a clear prefix.
        for text in summary_texts:
            rebuilt.append({
                "role": "system",
                "content": text,
                "name": "compression_summary",
            })

        rebuilt.extend(recent)

        # ── 6. Replace in place and track stats ─────────────────
        after_total = sum(_msg_tokens(m) for m in rebuilt)

        self.stats.total_tokens_before = before_total
        self.stats.total_tokens_after = after_total
        self.stats.messages_compressed = total_compressed
        self.stats.summary_count = len(summary_texts)
        self.stats.summaries = self._last_summaries

        messages.clear()
        messages.extend(rebuilt)
        return messages

    # ── Unit grouping ───────────────────────────────────────────

    def _group_into_units(self, messages: list[dict]) -> list[list[dict]]:
        """Group messages into units where tool-call + tool-result stay together.

        A *unit* starts on a user message or a standalone assistant message.
        Subsequent assistant tool_calls and tool results are appended to the
        same unit so they are summarised (or kept) as one atomic block.
        """
        if not messages:
            return []

        units: list[list[dict]] = []
        current: list[dict] = []

        for m in messages:
            role = m.get("role", "")

            # Start a new unit on user or standalone assistant (no tool calls).
            if role == "user":
                if current:
                    units.append(current)
                current = [m]
            elif role == "assistant" and not m.get("tool_calls"):
                if current:
                    units.append(current)
                current = [m]
            elif role in ("assistant", "tool"):
                # Belongs to current unit (tool call + result chain).
                current.append(m)
            else:
                # System or unrecognised — flush and start fresh.
                if current:
                    units.append(current)
                current = [m]

        if current:
            units.append(current)

        return units

    # ── Summarisation ───────────────────────────────────────────

    def _summarize_unit(self, unit: list[dict]) -> str:
        """Produce a summary for a single conversation unit."""
        if self._summarize_model:
            try:
                return self._llm_summarize_unit(unit)
            except Exception as e:
                logger.debug("LLM unit summarization failed, heuristic fallback: %s", e)
        return self._heuristic_summarize_unit(unit)

    def _llm_summarize_unit(self, unit: list[dict]) -> str:
        """Use the configured LLM to summarise a unit."""
        # Build a compact transcript.
        transcript_lines: list[str] = []
        for m in unit:
            role = m.get("role", "?")
            content = str(m.get("content", ""))
            if m.get("tool_calls"):
                tools = [
                    tc.get("function", {}).get("name", "?")
                    for tc in m["tool_calls"]
                ]
                transcript_lines.append(f"[{role} called: {', '.join(tools)}]")
            elif role == "tool":
                transcript_lines.append(f"[tool result ({m.get('name', '?')})]: {content[:200]}")
            elif content:
                transcript_lines.append(f"[{role}]: {content[:300]}")

        prompt = (
            "Summarize this conversation segment in 2-4 sentences. "
            "Include the user's request, any tool calls made, key results, "
            "and errors if any.\n\n"
            + "\n".join(transcript_lines)
        )

        response = self._summarize_model.chat(
            [{"role": "system", "content": self.SUMMARIZE_SYSTEM},
             {"role": "user", "content": prompt}],
            max_tokens=self.max_summary_tokens,
            temperature=0,
        )
        summary = response.choices[0].message.content
        return f"[SUMMARY: {summary}]"

    def _heuristic_summarize_unit(self, unit: list[dict]) -> str:
        """Heuristic summary — extract key snippets without an LLM call."""
        parts: list[str] = []

        for m in unit:
            role = m.get("role", "?")
            content = str(m.get("content", ""))

            if m.get("tool_calls"):
                tools = [
                    tc.get("function", {}).get("name", "?")
                    for tc in m["tool_calls"]
                ]
                parts.append(f"[{role} called: {', '.join(tools)}]")
            elif role == "tool":
                name = m.get("name", "?")
                snippet = content[:100].replace("\n", " ")
                parts.append(f"[tool {name}]: {snippet}")
            elif content:
                snippet = content[:150].replace("\n", " ")
                parts.append(f"[{role}]: {snippet}")

        joined = " → ".join(parts) if parts else "(empty turn)"
        return f"[SUMMARY: {joined}]"

    # ── Stats / reporting ───────────────────────────────────────

    @property
    def summaries(self) -> list[SummaryBlock]:
        return self._last_summaries

    @property
    def total_tokens_saved(self) -> int:
        return self.stats.total_tokens_saved

    @property
    def compression_ratio(self) -> float:
        return self.stats.compression_ratio

    def reset_stats(self) -> None:
        self.stats = CompressionStats()
        self._last_summaries = []

    def __repr__(self) -> str:
        return (
            f"TrajectoryCompressor(keep_recent={self.keep_recent}, "
            f"max_summary_tokens={self.max_summary_tokens}, "
            f"llm={self._summarize_model is not None})"
        )
