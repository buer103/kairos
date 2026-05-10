"""Memory middleware — tier-aware injection with confidence filtering.

v2 upgrade from flat MemoryBackend to TieredMemoryStore:
  - Per-agent isolation (agent_id dimension)
  - Three tiers: profile, timeline, facts
  - Confidence ≥ 0.7 filtering for fact injection
  - Max injection tokens budget (2000 by default)
  - DeerFlow-compatible format_for_prompt output
"""

from __future__ import annotations

import logging
from typing import Any

from kairos.core.middleware import Middleware
from kairos.memory.backends import MemoryBackend, SQLiteBackend
from kairos.memory.middleware import (
    _extract_facts,
    _extract_keywords,
    _extract_user_message,
)
from kairos.memory.tiers import (
    CONFIDENCE_THRESHOLD,
    MAX_INJECTION_TOKENS,
    MemoryTier,
    TieredMemoryStore,
)

logger = logging.getLogger("kairos.memory.middleware_v2")


class MemoryMiddlewareV2(Middleware):
    """Tier-aware memory injection with confidence filtering.

    Replaces the flat MemoryMiddleware with three-tier memory:
      - Profile: stable user attributes (overwrite)
      - Timeline: chronological events (append-only)
      - Facts: extracted with confidence ≥ 0.7 (TTL expiry)

    Parameters:
        tiered_store: TieredMemoryStore instance (created if not provided).
        agent_id: Agent identifier for per-agent isolation (default "default").
        max_injection_tokens: Token budget for memory injection (2000 default).
        auto_save_facts: Auto-extract and save facts from responses.
        fact_confidence_threshold: Minimum confidence to save a fact (0.7).
    """

    def __init__(
        self,
        tiered_store: TieredMemoryStore | None = None,
        *,
        agent_id: str = "default",
        max_injection_tokens: int = MAX_INJECTION_TOKENS,
        auto_save_facts: bool = True,
        fact_confidence_threshold: float = CONFIDENCE_THRESHOLD,
        # Backward compat
        backend: MemoryBackend | None = None,
        memory_store: Any = None,
        max_injected: int = 5,
        max_injected_chars: int = 2000,
        auto_save: bool = True,
    ) -> None:
        # Create or accept TieredMemoryStore
        if tiered_store:
            self._store = tiered_store
        elif backend:
            self._store = TieredMemoryStore(
                backend=backend,
                confidence_threshold=fact_confidence_threshold,
                max_injection_tokens=max_injection_tokens,
            )
        else:
            self._store = TieredMemoryStore(
                backend=SQLiteBackend(),
                confidence_threshold=fact_confidence_threshold,
                max_injection_tokens=max_injection_tokens,
            )

        self._agent_id = agent_id
        self._auto_save = auto_save_facts
        self._max_injected = max_injected
        self._max_injected_chars = max_injected_chars

    # ---- Properties ---------------------------------------------------

    @property
    def store(self) -> TieredMemoryStore:
        return self._store

    @property
    def agent_id(self) -> str:
        return self._agent_id

    # ---- Middleware hooks ---------------------------------------------

    def before_agent(
        self, state: Any, runtime: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Inject ALL tiers into system prompt at session start."""
        messages = getattr(state, "messages", [])
        if not messages:
            return None

        block = self._store.format_for_prompt(agent_id=self._agent_id)
        if not block:
            return None

        if messages[0].get("role") == "system":
            messages[0]["content"] = (
                messages[0].get("content", "") + "\n\n" + block
            )
        else:
            messages.insert(0, {"role": "system", "content": block})
        return None

    def after_agent(
        self, state: Any, runtime: dict[str, Any]
    ) -> dict[str, Any] | None:
        """No-op — before_model/after_model handle per-turn operations."""
        return None

    def before_model(
        self, state: Any, runtime: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Search tiered memory for context relevant to the latest user message."""
        messages = getattr(state, "messages", [])
        user_text = _extract_user_message(messages)
        if not user_text or not messages:
            return None

        keywords = _extract_keywords(user_text, max_keywords=5)
        if not keywords:
            return None

        query = " OR ".join(keywords)
        results = self._store.search(
            query,
            agent_id=self._agent_id,
            tiers=[MemoryTier.FACT, MemoryTier.PROFILE],
            limit=self._max_injected,
        )

        if not results:
            return None

        block = self._format_injected_block(results)
        if not block:
            return None

        if messages[0].get("role") == "system":
            messages[0]["content"] = (
                messages[0].get("content", "") + "\n\n" + block
            )
        else:
            messages.insert(0, {"role": "system", "content": block})
        return None

    def after_model(
        self, state: Any, runtime: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Extract and persist facts from the assistant response."""
        messages = getattr(state, "messages", [])
        if not self._auto_save:
            return None

        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                if isinstance(content, str) and content.strip():
                    facts = _extract_facts(content)
                    for fact_text, category in facts:
                        # Extract confidence from context if available
                        confidence = self._estimate_confidence(fact_text, content)
                        self._store.save_fact(
                            key=fact_text[:60],
                            value=fact_text,
                            confidence=confidence,
                            agent_id=self._agent_id,
                            category=category,
                        )
                break
        return None

    # ---- Public API --------------------------------------------------

    def save_profile(self, key: str, value: str) -> None:
        """Save a user profile attribute."""
        self._store.save_profile(key, value, agent_id=self._agent_id)

    def append_timeline(self, event_type: str, description: str) -> None:
        """Append a timeline event."""
        self._store.append_timeline(
            event_type, description, agent_id=self._agent_id
        )

    def save_fact(
        self, key: str, value: str, confidence: float, category: str = "fact"
    ) -> bool:
        """Save a fact with explicit confidence."""
        return self._store.save_fact(
            key, value, confidence, agent_id=self._agent_id, category=category
        )

    def search(self, query: str, limit: int = 20) -> list:
        """Search tiered memory."""
        return self._store.search(query, agent_id=self._agent_id, limit=limit)

    def format_prompt(self, max_tokens: int | None = None) -> str:
        """Format memory for prompt injection."""
        return self._store.format_for_prompt(
            agent_id=self._agent_id, max_tokens=max_tokens
        )

    def stats(self) -> dict:
        """Return memory statistics."""
        return self._store.stats(agent_id=self._agent_id)

    # ---- Internal ----------------------------------------------------

    def _estimate_confidence(
        self, fact_text: str, full_response: str, default: float = 0.75
    ) -> float:
        """Estimate confidence for an extracted fact.

        Uses heuristics:
          - Explicit "I'm certain" / "definitely" → +0.15
          - Hedging "might" / "possibly" / "perhaps" → -0.2
          - Short vague fact → -0.1
          - Fact appears near "confidence: X" → use X
        """
        confidence = default
        lowered = full_response.lower()
        context = fact_text.lower()

        # Confidence keywords
        if any(w in context for w in ("definitely", "certainly", "always", "confirmed")):
            confidence += 0.15
        if any(w in context for w in ("might", "may", "possibly", "perhaps", "maybe", "could be")):
            confidence -= 0.2
        if len(fact_text.split()) < 4:
            confidence -= 0.1

        return max(0.0, min(1.0, confidence))

    def _format_injected_block(
        self, entries: list, header: str = "## RELEVANT MEMORY"
    ) -> str:
        """Format tiered search results for system prompt injection."""
        lines = [header]
        total_chars = len(header) + 1

        for entry in entries:
            if hasattr(entry, "to_dict"):
                d = entry.to_dict()
            else:
                d = entry

            tier = d.get("tier", "")
            cat = d.get("category", "")
            conf = d.get("confidence", 1.0)

            tag_parts = []
            if tier:
                tag_parts.append(f"[{tier}]")
            if cat:
                tag_parts.append(f"[{cat}]")
            if conf < 1.0:
                tag_parts.append(f"(conf:{conf:.2f})")
            tag = " ".join(tag_parts) + " " if tag_parts else ""

            key = d.get("key", "")
            value = d.get("value", "")
            line = f"- {tag}**{key}**: {value}"

            if total_chars + len(line) > self._max_injected_chars:
                break
            lines.append(line)
            total_chars += len(line)

        return "\n".join(lines) if len(lines) > 1 else ""
