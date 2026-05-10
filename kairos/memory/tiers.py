"""Three-tier memory system — DeerFlow-inspired profile/timeline/facts.

DeerFlow's memory architecture:
  **画像 (Profile)** — stable user attributes, overwrite-on-update
  **时间线 (Timeline)** — chronological event records, append-only
  **Facts** — extracted facts, confidence ≥ 0.7, 5 categories

Differences from flat MemoryStore:
  - Per-agent isolation (agent_id dimension)
  - Tier-aware save/load/search with different semantics per tier
  - Confidence threshold for fact storage (rejects < 0.7)
  - Max injection tokens budget
  - Fact expiry (TTL-based staleness)
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from kairos.memory.backends import MemoryBackend, SQLiteBackend

logger = logging.getLogger("kairos.memory.tiers")

# ============================================================================
# Constants
# ============================================================================

CONFIDENCE_THRESHOLD = 0.7  # DeerFlow default
MAX_INJECTION_TOKENS = 2000  # DeerFlow default
DEFAULT_TTL_DAYS = 90  # Facts expire after 90 days

FACT_CATEGORIES = frozenset({
    "preference",   # User likes/dislikes
    "fact",         # Objective fact about the world/user
    "knowledge",    # Domain knowledge
    "decision",     # User decisions / choices
    "action",       # Actions taken
})


class MemoryTier(str, Enum):
    """Three-tier memory with distinct semantics."""
    PROFILE = "profile"    # Stable user attributes — overwrite semantics
    TIMELINE = "timeline"  # Chronological events — append-only
    FACT = "fact"          # Extracted facts — confidence≥0.7, TTL


# ============================================================================
# Data classes
# ============================================================================


@dataclass
class MemoryEntry:
    """A single memory entry in any tier."""

    key: str
    value: str
    tier: MemoryTier
    agent_id: str = "default"
    category: str = ""
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = 0.0
    updated_at: float = 0.0
    ttl: float | None = None  # seconds, None = no expiry

    def is_expired(self, now: float | None = None) -> bool:
        """Check if this entry has passed its TTL."""
        if self.ttl is None:
            return False
        now = now or time.time()
        return (now - self.created_at) > self.ttl

    def meets_confidence(self, threshold: float = CONFIDENCE_THRESHOLD) -> bool:
        """Check if this entry meets the minimum confidence."""
        return self.confidence >= threshold

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "value": self.value,
            "tier": self.tier.value,
            "agent_id": self.agent_id,
            "category": self.category,
            "confidence": self.confidence,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "ttl": self.ttl,
        }


# ============================================================================
# Confidence filter
# ============================================================================


class ConfidenceFilter:
    """Filter facts by confidence threshold. DeerFlow default: 0.7."""

    def __init__(self, threshold: float = CONFIDENCE_THRESHOLD):
        self.threshold = threshold

    def passes(self, confidence: float) -> bool:
        return confidence >= self.threshold

    def filter(self, entries: list[MemoryEntry]) -> list[MemoryEntry]:
        return [e for e in entries if e.meets_confidence(self.threshold)]


# ============================================================================
# Tiered Memory Store
# ============================================================================


class TieredMemoryStore:
    """Three-tier persistent memory with per-agent isolation.

    Backed by SQLiteBackend. Implements DeerFlow-style profile/timeline/facts
    with confidence filtering and max injection token budget.

    Example::

        store = TieredMemoryStore()
        store.save_profile("lang", "zh-CN", agent_id="buer")
        store.append_timeline("code_review", "Reviewed kairos PR #42", agent_id="buer")
        store.save_fact("kairos version", "v0.15.0-dev", confidence=0.9,
                        category="fact", agent_id="buer")

        # Inject into prompt (respects token budget)
        block = store.format_for_prompt(agent_id="buer", max_tokens=2000)
    """

    def __init__(
        self,
        backend: MemoryBackend | None = None,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        max_injection_tokens: int = MAX_INJECTION_TOKENS,
        fact_ttl_days: int = DEFAULT_TTL_DAYS,
    ):
        self._backend = backend or SQLiteBackend()
        self._confidence = ConfidenceFilter(confidence_threshold)
        self._max_injection_tokens = max_injection_tokens
        self._fact_ttl = fact_ttl_days * 86400 if fact_ttl_days else None

    # ---- Profile (overwrite semantics) -----------------------------------

    def save_profile(
        self,
        key: str,
        value: str,
        agent_id: str = "default",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Save a profile attribute. Overwrites if key exists."""
        self._save(
            key=f"profile:{agent_id}:{key}",
            value=value,
            tier=MemoryTier.PROFILE,
            agent_id=agent_id,
            confidence=1.0,
            metadata=metadata,
        )

    def get_profile(self, key: str, agent_id: str = "default") -> MemoryEntry | None:
        return self._load(f"profile:{agent_id}:{key}", agent_id, MemoryTier.PROFILE)

    def list_profiles(self, agent_id: str = "default") -> list[MemoryEntry]:
        return self._list_by_tier(MemoryTier.PROFILE, agent_id)

    # ---- Timeline (append-only) ------------------------------------------

    def append_timeline(
        self,
        event_type: str,
        description: str,
        agent_id: str = "default",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Append a chronological event. Each call creates a new entry.

        The key is auto-generated from timestamp + event_type to ensure
        append-only semantics (never overwrites).
        """
        ts = int(time.time() * 1000)
        key = f"timeline:{event_type}:{ts}"
        self._save(
            key=key,
            value=description,
            tier=MemoryTier.TIMELINE,
            agent_id=agent_id,
            confidence=1.0,
            metadata=metadata,
        )

    def get_timeline(
        self,
        agent_id: str = "default",
        event_type: str | None = None,
        limit: int = 50,
    ) -> list[MemoryEntry]:
        """Get timeline events, newest first."""
        entries = self._list_by_tier(MemoryTier.TIMELINE, agent_id)
        if event_type:
            entries = [
                e for e in entries
                if e.key.startswith(f"timeline:{event_type}:")
            ]
        # Sort by creation time descending
        entries.sort(key=lambda e: e.created_at, reverse=True)
        return entries[:limit]

    # ---- Facts (confidence-filtered) -------------------------------------

    def save_fact(
        self,
        key: str,
        value: str,
        confidence: float,
        agent_id: str = "default",
        category: str = "fact",
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Save a fact. Returns False if confidence < threshold (rejected).

        Facts use TTL — automatically expire after fact_ttl_days.
        """
        if confidence < self._confidence.threshold:
            logger.debug(
                "Fact rejected: confidence %.2f < threshold %.2f: %s",
                confidence, self._confidence.threshold, key,
            )
            return False

        if category not in FACT_CATEGORIES:
            logger.warning("Unknown fact category '%s', using 'fact'", category)
            category = "fact"

        self._save(
            key=f"fact:{agent_id}:{key}",
            value=value,
            tier=MemoryTier.FACT,
            agent_id=agent_id,
            category=category,
            confidence=confidence,
            metadata=metadata,
            ttl=self._fact_ttl,
        )
        return True

    def get_fact(self, key: str, agent_id: str = "default") -> MemoryEntry | None:
        return self._load(f"fact:{agent_id}:{key}", agent_id, MemoryTier.FACT)

    def list_facts(
        self,
        agent_id: str = "default",
        category: str | None = None,
        min_confidence: float = CONFIDENCE_THRESHOLD,
    ) -> list[MemoryEntry]:
        """List facts, filtered by confidence and optional category."""
        entries = self._list_by_tier(MemoryTier.FACT, agent_id)
        if category:
            entries = [e for e in entries if e.category == category]
        entries = [e for e in entries if e.confidence >= min_confidence]
        entries.sort(key=lambda e: e.confidence, reverse=True)
        return entries

    # ---- Search across tiers ---------------------------------------------

    def search(
        self,
        query: str,
        agent_id: str = "default",
        tiers: list[MemoryTier] | None = None,
        limit: int = 20,
    ) -> list[MemoryEntry]:
        """Search all (or specified) tiers. Facts are confidence-filtered."""
        raw = self._backend.search(query, limit=limit * 2)  # Extra for filtering

        results: list[MemoryEntry] = []
        for r in raw:
            entry = self._row_to_entry(r)
            if entry.agent_id and entry.agent_id != agent_id:
                continue
            if tiers and entry.tier not in tiers:
                continue
            if entry.tier == MemoryTier.FACT and not entry.meets_confidence(
                self._confidence.threshold
            ):
                continue
            results.append(entry)

        return results[:limit]

    # ---- Prompt formatting -----------------------------------------------

    def format_for_prompt(
        self,
        agent_id: str = "default",
        max_tokens: int | None = None,
        include_profile: bool = True,
        include_timeline: bool = True,
        include_facts: bool = True,
    ) -> str:
        """Format memory as a prompt block, respecting token budget.

        Returns empty string if nothing to inject. Applies DeerFlow's
        max_injection_tokens budget (default 2000) with tier priority:
        PROFILE > FACTS > TIMELINE.
        """
        max_tokens = max_tokens or self._max_injection_tokens
        # Rough estimate: 1 token ≈ 4 chars for English/Chinese mix
        max_chars = max_tokens * 3

        sections: list[str] = []
        chars_used = 0

        if include_profile:
            profiles = self.list_profiles(agent_id)
            if profiles:
                block = "## USER PROFILE\n" + "\n".join(
                    f"- {e.key.split(':', 2)[-1]}: {e.value}" for e in profiles
                )
                if chars_used + len(block) <= max_chars:
                    sections.append(block)
                    chars_used += len(block)

        if include_facts:
            facts = self.list_facts(agent_id)
            if facts:
                block = "## FACTS\n" + "\n".join(
                    f"- [{e.category}] {e.key.split(':', 2)[-1]}: {e.value} (confidence:{e.confidence:.2f})"
                    for e in facts
                )
                available = max_chars - chars_used
                if available > 0:
                    if len(block) > available:
                        block = block[:available] + "\n..."
                    sections.append(block)
                    chars_used += len(block)

        if include_timeline:
            timeline = self.get_timeline(agent_id, limit=10)
            if timeline:
                block = "## TIMELINE\n" + "\n".join(
                    f"- [{e.key.split(':')[1]}] {e.value}" for e in timeline
                )
                available = max_chars - chars_used
                if available > 0:
                    if len(block) > available:
                        block = block[:available] + "\n..."
                    sections.append(block)

        return "\n\n".join(sections) if sections else ""

    # ---- Per-agent management --------------------------------------------

    def clear_agent(self, agent_id: str) -> int:
        """Clear all memory for a specific agent. Returns count deleted."""
        count = 0
        for tier in MemoryTier:
            entries = self._list_by_tier(tier, agent_id)
            for e in entries:
                self._backend.delete(e.key)
                count += 1
        return count

    def stats(self, agent_id: str = "default") -> dict[str, Any]:
        """Return memory statistics for an agent."""
        return {
            "profile_count": len(self.list_profiles(agent_id)),
            "timeline_count": len(self.get_timeline(agent_id)),
            "fact_count": len(self.list_facts(agent_id, min_confidence=0)),
            "fact_above_threshold": len(self.list_facts(agent_id)),
            "confidence_threshold": self._confidence.threshold,
            "max_injection_tokens": self._max_injection_tokens,
        }

    # ---- Internal --------------------------------------------------------

    def _save(
        self,
        key: str,
        value: str,
        tier: MemoryTier,
        agent_id: str = "default",
        confidence: float = 1.0,
        category: str = "",
        metadata: dict[str, Any] | None = None,
        ttl: float | None = None,
    ) -> None:
        now = time.time()
        # For PROFILE tier: overwrite (backend.save already does upsert)
        # For TIMELINE tier: key is unique per-timestamp, never overwrites
        # For FACT tier: overwrite if same key exists
        self._backend.save(
            key=key,
            value=value,
            category=f"{tier.value}:{agent_id}:{category}" if category else f"{tier.value}:{agent_id}",
            metadata={
                **(metadata or {}),
                "tier": tier.value,
                "agent_id": agent_id,
                "confidence": confidence,
                "ttl": ttl,
            },
            ttl=ttl,
        )

    def _load(self, key: str, agent_id: str, tier: MemoryTier) -> MemoryEntry | None:
        raw = self._backend.load(key)
        if raw is None:
            return None
        entry = self._row_to_entry(raw)
        if entry.agent_id != agent_id or entry.tier != tier:
            return None
        if entry.is_expired():
            self._backend.delete(key)
            return None
        return entry

    def _list_by_tier(
        self, tier: MemoryTier, agent_id: str = "default"
    ) -> list[MemoryEntry]:
        keys = self._backend.list_keys()
        entries: list[MemoryEntry] = []
        now = time.time()
        prefix = f"{tier.value}:"
        for key in keys:
            raw = self._backend.load(key)
            if raw is None:
                continue
            entry = self._row_to_entry(raw)
            if entry.agent_id != agent_id or entry.tier != tier:
                continue
            # Purge expired facts
            if entry.is_expired(now):
                self._backend.delete(key)
                continue
            entries.append(entry)
        return entries

    def _row_to_entry(self, raw: dict[str, Any]) -> MemoryEntry:
        """Convert backend row dict to MemoryEntry."""
        import json as _json

        # metadata is stored as JSON string by the backend
        meta_raw = raw.get("metadata", {})
        if isinstance(meta_raw, str):
            try:
                meta = _json.loads(meta_raw)
            except (json.JSONDecodeError, TypeError):
                meta = {}
        elif isinstance(meta_raw, dict):
            meta = meta_raw
        else:
            meta = {}

        category_raw = raw.get("category", "")
        # category format: "tier:agent_id:category" or "tier:agent_id"
        parts = category_raw.split(":", 2)
        tier_str = parts[0] if parts else ""
        agent_id_from_cat = parts[1] if len(parts) > 1 else "default"
        cat = parts[2] if len(parts) > 2 else ""

        try:
            tier = MemoryTier(tier_str)
        except ValueError:
            tier = MemoryTier.FACT

        def _to_float(v: Any) -> float:
            if isinstance(v, (int, float)):
                return float(v)
            try:
                return float(v)
            except (TypeError, ValueError):
                return time.time()

        return MemoryEntry(
            key=raw.get("key", ""),
            value=raw.get("value", ""),
            tier=tier,
            agent_id=agent_id_from_cat,
            category=cat,
            confidence=meta.get("confidence", 1.0),
            metadata=meta,
            created_at=_to_float(raw.get("created_at", 0)),
            updated_at=_to_float(raw.get("updated_at", 0)),
            ttl=meta.get("ttl"),
        )
