"""Memory middleware — injects context before each model call, saves facts after.

Hooks:
- ``before_model`` — searches memory with keywords extracted from the
  latest user message, injecting relevant entries into the system prompt.
- ``after_model`` — extracts key facts from the assistant response and
  persists them (categories: preference, fact, conversation, project).

Uses the ``MemoryBackend`` abstraction so the storage engine
(SQLite + FTS5 or in-memory dict) is pluggable.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from typing import Any

from kairos.core.middleware import Middleware
from kairos.memory.backends import DictBackend, MemoryBackend, SQLiteBackend

logger = logging.getLogger("kairos.memory.middleware")

# ── Category constants ──────────────────────────────────────────────

CATEGORY_PREFERENCE = "preference"
CATEGORY_FACT = "fact"
CATEGORY_CONVERSATION = "conversation"
CATEGORY_PROJECT = "project"

VALID_CATEGORIES = frozenset({
    CATEGORY_PREFERENCE,
    CATEGORY_FACT,
    CATEGORY_CONVERSATION,
    CATEGORY_PROJECT,
})

# ── Simple keyword extraction ───────────────────────────────────────

# Stop-words to filter out when extracting keywords from user messages.
_STOP_WORDS: frozenset[str] = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves", "he", "him", "his",
    "himself", "she", "her", "hers", "herself", "it", "its", "itself",
    "they", "them", "their", "theirs", "themselves", "what", "which",
    "who", "whom", "this", "that", "these", "those", "am", "at", "by",
    "for", "with", "about", "between", "through", "during", "before",
    "after", "above", "below", "from", "up", "down", "in", "out", "on",
    "off", "over", "under", "again", "further", "then", "once", "here",
    "there", "when", "where", "why", "how", "all", "both", "each", "few",
    "more", "most", "other", "some", "such", "no", "nor", "not", "only",
    "own", "same", "so", "than", "too", "very", "just", "because", "as",
    "until", "while", "if", "or", "and", "but", "to", "of", "also",
})


def _extract_keywords(text: str, max_keywords: int = 5) -> list[str]:
    """Extract meaningful lowercase keywords from a user message.

    Strips punctuation, removes stop-words, and returns the longest
    remaining tokens first (up to *max_keywords*).
    """
    tokens = re.findall(r"[a-zA-Z0-9_]{3,}", text.lower())
    meaningful = [t for t in tokens if t not in _STOP_WORDS]
    # Prefer longer, more specific tokens.
    meaningful.sort(key=len, reverse=True)
    seen: set[str] = set()
    result: list[str] = []
    for t in meaningful:
        if t not in seen:
            seen.add(t)
            result.append(t)
        if len(result) >= max_keywords:
            break
    return result


def _extract_user_message(messages: list[dict[str, Any]]) -> str | None:
    """Return the content of the last ``user`` message, if any."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            # Handle list-of-blocks content (e.g. multimodal).
            if isinstance(content, list):
                return " ".join(
                    b.get("text", "") if isinstance(b, dict) else str(b)
                    for b in content
                )
    return None


# ── Fact extraction from assistant responses ────────────────────────

_FACT_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # "User prefers X", "I prefer X"
    (re.compile(r"(?:user|you|i)\s+(?:prefers?|like|love|enjoy|dislike|hate|want)\s+(.+?)[.!]", re.I),
     CATEGORY_PREFERENCE),
    # "The project uses X", "Project X is Y"
    (re.compile(r"(?:the\s+)?project\s+(?:uses?|is|has|requires?|needs?)\s+(.+?)[.!]", re.I),
     CATEGORY_PROJECT),
    # "Remember that X", "Note: X"
    (re.compile(r"(?:remember\s+that|note|important|key\s+fact)\s*[:;]\s*(.+?)[.!]", re.I),
     CATEGORY_FACT),
]


def _extract_facts(text: str) -> list[tuple[str, str]]:
    """Extract (fact_text, category) tuples from an assistant response.

    Uses simple regex patterns; a future version could use an LLM call.
    """
    facts: list[tuple[str, str]] = []
    for pattern, category in _FACT_PATTERNS:
        for match in pattern.finditer(text):
            fact_text = match.group(1).strip()
            if 3 <= len(fact_text) <= 300:
                facts.append((fact_text, category))
    # Deduplicate while preserving order.
    seen: set[str] = set()
    unique: list[tuple[str, str]] = []
    for f, c in facts:
        if f.lower() not in seen:
            seen.add(f.lower())
            unique.append((f, c))
    return unique


# ── Middleware ──────────────────────────────────────────────────────


class MemoryMiddleware(Middleware):
    """Inject relevant memories before each model call; save facts after.

    Parameters:
        backend: A :class:`MemoryBackend` instance.  Defaults to
                 :class:`SQLiteBackend` (``~/.kairos/memory/memory.db``).
        max_injected: Maximum number of memories to inject into the
                      system prompt (default 5).
        max_injected_chars: Character budget for the injected block
                            (default 2000).
        auto_save: Whether to auto-extract and save facts from
                   assistant responses (default ``True``).
    """

    def __init__(
        self,
        backend: MemoryBackend | None = None,
        *,
        memory_store: Any = None,  # deprecated — kept for backward compat
        max_injected: int = 5,
        max_injected_chars: int = 2000,
        auto_save: bool = True,
    ) -> None:
        if memory_store is not None:
            import warnings
            warnings.warn(
                "memory_store is deprecated; pass backend= instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        if backend is not None:
            self._backend: MemoryBackend = backend
        elif memory_store is not None:
            # Adapt legacy MemoryStore to MemoryBackend protocol.
            self._backend = _LegacyStoreAdapter(memory_store)
        else:
            self._backend = SQLiteBackend()
        self._max_injected = max_injected
        self._max_injected_chars = max_injected_chars
        self._auto_save = auto_save
        # Pending writes queued via queue_write() — flushed in after_model.
        self._pending_writes: list[dict[str, Any]] = []

    # ── Middleware hooks ────────────────────────────────────────────

    def before_agent(
        self, state: Any, runtime: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Compatibility hook — injects *all* recent memories into the system
        prompt (broad injection).  Prefer :meth:`before_model` for targeted
        keyword-based injection.
        """
        messages: list[dict[str, Any]] = getattr(state, "messages", [])
        if not messages:
            return None

        # Load all memories (broad injection like legacy behaviour).
        keys = self._backend.list_keys()
        if not keys:
            return None

        results: list[dict[str, Any]] = []
        chars = 0
        for key in keys:
            entry = self._backend.load(key)
            if entry is None:
                continue
            line_len = len(entry.get("key", "")) + len(entry.get("value", ""))
            if chars + line_len > self._max_injected_chars:
                break
            results.append(entry)
            chars += line_len

        block = self._format_injected_block(results, header="## MEMORY (persistent knowledge)")
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
        """Compatibility hook — flush pending writes at session end."""
        self._flush_pending()
        return None

    def before_model(
        self, state: Any, runtime: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Search memory for context relevant to the user's latest message
        and inject it into the system prompt.
        """
        messages: list[dict[str, Any]] = getattr(state, "messages", [])
        user_text = _extract_user_message(messages)
        if not user_text or not messages:
            return None

        # Build a search query from extracted keywords.
        keywords = _extract_keywords(user_text, max_keywords=5)
        if not keywords:
            return None

        query = " OR ".join(keywords)
        results = self._backend.search(query, limit=self._max_injected)

        if not results:
            return None

        block = self._format_injected_block(results)
        if not block:
            return None

        # Inject into the system message (first message if it has role=system).
        if messages[0].get("role") == "system":
            separator = "\n\n" if messages[0].get("content", "").endswith("\n") else "\n\n"
            messages[0]["content"] = (
                messages[0].get("content", "") + separator + block
            )
        else:
            # Prepend a synthetic system message.
            messages.insert(0, {"role": "system", "content": block})

        return None

    def after_model(
        self, state: Any, runtime: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Extract and persist facts from the assistant response, then flush
        any pending writes queued via :meth:`queue_write`.
        """
        messages: list[dict[str, Any]] = getattr(state, "messages", [])

        if self._auto_save:
            # Look at the last assistant message.
            for msg in reversed(messages):
                if msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    if isinstance(content, str) and content.strip():
                        facts = _extract_facts(content)
                        for fact_text, category in facts:
                            key = _make_fact_key(fact_text)
                            self._backend.save(
                                key, fact_text, category=category,
                            )
                    break

        # Flush pending writes.
        self._flush_pending()
        return None

    # ── Public API ──────────────────────────────────────────────────

    def search_memory(
        self, query: str, *, limit: int = 20, category: str | None = None
    ) -> list[dict[str, Any]]:
        """Search the memory backend directly."""
        return self._backend.search(query, limit=limit, category=category)

    def save_memory(
        self,
        key: str,
        value: str,
        *,
        category: str = CATEGORY_FACT,
        metadata: dict[str, Any] | None = None,
        ttl: float | None = None,
    ) -> None:
        """Persist a memory entry immediately."""
        self._backend.save(key, value, category=category, metadata=metadata, ttl=ttl)

    def load_memory(self, key: str) -> dict[str, Any] | None:
        """Load a single memory entry."""
        return self._backend.load(key)

    def delete_memory(self, key: str) -> bool:
        """Delete a memory entry."""
        return self._backend.delete(key)

    def list_memory_keys(
        self, prefix: str = "", *, category: str | None = None
    ) -> list[str]:
        """List stored memory keys."""
        return self._backend.list_keys(prefix, category=category)

    def clear_memory(self, *, category: str | None = None) -> int:
        """Clear all (or category-filtered) memories."""
        return self._backend.clear(category=category)

    def queue_write(
        self,
        key: str,
        value: str,
        *,
        category: str = CATEGORY_FACT,
        metadata: dict[str, Any] | None = None,
        ttl: float | None = None,
    ) -> None:
        """Queue a write to be persisted at session end (in ``after_model``).

        Useful for tool-driven memory operations that should be batched.
        """
        self._pending_writes.append({
            "key": key,
            "value": value,
            "category": category,
            "metadata": metadata,
            "ttl": ttl,
        })

    # ── Properties ──────────────────────────────────────────────────

    @property
    def backend(self) -> MemoryBackend:
        """The underlying memory backend."""
        return self._backend

    # ── Internal ────────────────────────────────────────────────────

    def _flush_pending(self) -> None:
        """Persist all queued writes, swallowing individual errors."""
        for write in self._pending_writes:
            try:
                self._backend.save(
                    key=write["key"],
                    value=write["value"],
                    category=write.get("category", CATEGORY_FACT),
                    metadata=write.get("metadata"),
                    ttl=write.get("ttl"),
                )
            except Exception:
                logger.exception(
                    "Failed to flush pending memory write for key=%s",
                    write.get("key"),
                )
        self._pending_writes.clear()

    def _format_injected_block(
        self, results: list[dict[str, Any]], header: str = "## RELEVANT MEMORY"
    ) -> str:
        """Format search results as a compact memory block for the system prompt."""
        lines: list[str] = []
        total_chars = 0
        total_chars += len(header) + 1

        for entry in results:
            cat = entry.get("category", "")
            cat_tag = f"[{cat}] " if cat else ""
            line = f"- {cat_tag}**{entry['key']}**: {entry['value']}"
            if total_chars + len(line) > self._max_injected_chars:
                break
            lines.append(line)
            total_chars += len(line)

        if not lines:
            return ""
        return header + "\n" + "\n".join(lines)


# ── Helpers ─────────────────────────────────────────────────────────


def _make_fact_key(fact_text: str) -> str:
    """Derive a stable key from fact text (short hash of normalised text)."""
    normalised = re.sub(r"\s+", " ", fact_text.lower()).strip()
    digest = hashlib.sha256(normalised.encode()).hexdigest()[:12]
    return f"fact:{digest}"


# ── Legacy adapter ──────────────────────────────────────────────────


class _LegacyStoreAdapter(MemoryBackend):
    """Wraps an old ``MemoryStore`` so it quacks like a ``MemoryBackend``.

    Only the subset of operations needed by ``MemoryMiddleware`` is mapped.
    For full-text search this falls back to the legacy ``LIKE``-based search.
    """

    def __init__(self, store: Any) -> None:
        self._store = store

    def save(
        self,
        key: str,
        value: str,
        *,
        category: str = "",
        metadata: dict[str, Any] | None = None,
        ttl: float | None = None,
    ) -> None:
        # Legacy MemoryStore uses scope for grouping; map category → scope.
        scope = category if category else "memory"
        self._store.add(scope=scope, key=key, value=value)

    def load(self, key: str) -> dict[str, Any] | None:
        # Legacy store.get requires scope — try common scopes.
        for scope in ("memory", "user", "conversation", "preference", ""):
            val = self._store.get(scope, key)
            if val is not None:
                return {"key": key, "value": val, "category": scope}
        return None

    def delete(self, key: str) -> bool:
        for scope in ("memory", "user", "conversation", "preference", ""):
            if self._store.remove(scope, key):
                return True
        return False

    def search(
        self, query: str, *, limit: int = 20, category: str | None = None
    ) -> list[dict[str, Any]]:
        results = self._store.search(query, scope=category, limit=limit)
        return [
            {"key": r["key"], "value": r["value"], "category": r.get("scope", "")}
            for r in results
        ]

    def list_keys(
        self, prefix: str = "", *, category: str | None = None
    ) -> list[str]:
        raw = self._store.all(scope=category)
        keys = [r["key"] for r in raw]
        if prefix:
            keys = [k for k in keys if k.startswith(prefix)]
        return sorted(keys)

    def clear(self, *, category: str | None = None) -> int:
        return self._store.clear(scope=category)
