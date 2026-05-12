"""MemoryRouter — multi-backend memory routing with semantic + keyword fallback.

Routes queries to the best backend based on query characteristics:
- Short/phrase queries → vector (semantic) backend
- Long/technical queries → FTS5 (keyword) backend
- Merges results from multiple backends with deduplication.

Usage::

    from kairos.memory import MemoryRouter, SQLiteBackend, VectorMemoryBackend

    router = MemoryRouter(
        keyword_backend=SQLiteBackend(),
        vector_backend=VectorMemoryBackend(),
    )
    router.save("pref_lang", "User prefers Chinese", category="preference")
    results = router.search("language preference")  # routes to vector
    results = router.search("pref_lang")             # routes to keyword
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from kairos.memory.backends import MemoryBackend

logger = logging.getLogger("kairos.memory.router")


class MemoryRouter(MemoryBackend):
    """Route memory operations across multiple backends.

    Write path: saves to ALL backends (eventual consistency).
    Read path: routes ``search()`` to the best backend based on query type.
    ``load()`` reads from primary backend first, falls back to secondary.

    Config:
        keyword_backend: MemoryBackend — FTS5/BM25 backend (e.g. SQLiteBackend)
        vector_backend: MemoryBackend | None — semantic backend (e.g. VectorMemoryBackend)
        primary: str — which backend handles ``load()``/``delete()``. Options:
                  "keyword" (default), "vector", or "all"
        search_strategy: str — "auto" (default), "merge", "vector_only", "keyword_only"
        merge_dedup_threshold: float — Jaccard similarity for merge dedup (0.0-1.0)
    """

    def __init__(
        self,
        keyword_backend: MemoryBackend,
        vector_backend: MemoryBackend | None = None,
        *,
        primary: str = "keyword",
        search_strategy: str = "auto",
        merge_dedup_threshold: float = 0.7,
    ):
        self._keyword = keyword_backend
        self._vector = vector_backend
        self._primary = primary
        self._search_strategy = search_strategy
        self._merge_dedup_threshold = merge_dedup_threshold
        self._lock = threading.Lock()

    @property
    def keyword_backend(self) -> MemoryBackend:
        return self._keyword

    @property
    def vector_backend(self) -> MemoryBackend | None:
        return self._vector

    # ── MemoryBackend interface ──────────────────────────────────────

    def save(
        self,
        key: str,
        value: str,
        *,
        category: str = "",
        metadata: dict[str, Any] | None = None,
        ttl: float | None = None,
    ) -> None:
        """Save to all backends."""
        with self._lock:
            self._keyword.save(key, value, category=category, metadata=metadata, ttl=ttl)
            if self._vector:
                self._vector.save(key, value, category=category, metadata=metadata, ttl=ttl)

    def load(self, key: str) -> dict[str, Any] | None:
        result = self._keyword.load(key)
        if result is not None:
            return result
        if self._vector and self._primary in ("vector", "all"):
            return self._vector.load(key)
        return None

    def delete(self, key: str) -> bool:
        deleted = False
        with self._lock:
            if self._keyword.delete(key):
                deleted = True
            if self._vector:
                if self._vector.delete(key):
                    deleted = True
        return deleted

    def search(
        self, query: str, *, limit: int = 20, category: str | None = None
    ) -> list[dict[str, Any]]:
        strategy = self._resolve_strategy(query)

        if strategy == "keyword_only":
            return self._keyword.search(query, limit=limit, category=category)
        elif strategy == "vector_only":
            if self._vector:
                return self._vector.search(query, limit=limit, category=category)
            return self._keyword.search(query, limit=limit, category=category)
        elif strategy == "merge":
            kw_results = self._keyword.search(query, limit=limit, category=category)
            if self._vector:
                vec_results = self._vector.search(query, limit=limit, category=category)
                return self._merge_results(kw_results, vec_results, limit)
            return kw_results
        else:
            return self._keyword.search(query, limit=limit, category=category)

    # ── Strategy resolution ──────────────────────────────────────────

    def _resolve_strategy(self, query: str) -> str:
        """Determine which search strategy to use."""
        if self._search_strategy != "auto":
            return self._search_strategy

        # Auto: use query characteristics
        words = query.split()
        word_count = len(words)

        # Short, conceptual queries → vector
        if word_count <= 3 and not any(
            op in query for op in ("==", "!=", "<=", ">=", "AND", "OR", "NOT")
        ):
            if self._vector:
                return "merge"  # merge for short queries

        # Long, keyword-heavy → keyword
        return "keyword_only"

    # ── Result merging ───────────────────────────────────────────────

    def _merge_results(
        self,
        keyword_results: list[dict[str, Any]],
        vector_results: list[dict[str, Any]],
        limit: int,
    ) -> list[dict[str, Any]]:
        """Merge and deduplicate results from two backends.

        Keyword results get higher priority (ranked first).
        Vector results deduplicated by key overlap.
        """
        seen_keys: set[str] = set()
        merged: list[dict[str, Any]] = []

        # Keyword results first (higher precision)
        for r in keyword_results:
            key = r.get("key", r.get("value", ""))
            if key not in seen_keys:
                seen_keys.add(key)
                r["_source"] = "keyword"
                merged.append(r)

        # Add non-overlapping vector results
        for r in vector_results:
            key = r.get("key", r.get("value", ""))
            if key not in seen_keys:
                # Check for near-duplicate values
                is_dup = False
                for seen in merged:
                    if self._text_similarity(
                        r.get("value", ""), seen.get("value", "")
                    ) > self._merge_dedup_threshold:
                        is_dup = True
                        break
                if not is_dup:
                    seen_keys.add(key)
                    r["_source"] = "vector"
                    merged.append(r)

        return merged[:limit]

    @staticmethod
    def _text_similarity(a: str, b: str) -> float:
        """Simple Jaccard token similarity."""
        if not a or not b:
            return 0.0
        a_tokens = set(a.lower().split())
        b_tokens = set(b.lower().split())
        if not a_tokens or not b_tokens:
            return 0.0
        intersection = a_tokens & b_tokens
        union = a_tokens | b_tokens
        return len(intersection) / len(union)

    def count(self) -> dict[str, int]:
        """Return count per backend."""
        counts: dict[str, int] = {"keyword": self._keyword.count() if hasattr(self._keyword, "count") else 0}
        if self._vector and hasattr(self._vector, "count"):
            counts["vector"] = self._vector.count()
        return counts

    def list_keys(
        self, prefix: str = "", *, category: str | None = None
    ) -> list[str]:
        """List all keys across backends (deduplicated)."""
        keys: set[str] = set()
        try:
            keys.update(self._keyword.list_keys(prefix=prefix, category=category))
        except Exception:
            pass
        if self._vector:
            try:
                keys.update(self._vector.list_keys(prefix=prefix, category=category))
            except Exception:
                pass
        return sorted(keys)

    def clear(self, *, category: str | None = None) -> int:
        """Remove all (or category-filtered) memories from all backends."""
        total = 0
        with self._lock:
            total += self._keyword.clear(category=category) if hasattr(self._keyword, "clear") else 0
            if self._vector and hasattr(self._vector, "clear"):
                total += self._vector.clear(category=category)
        return total

    def __repr__(self) -> str:
        return (
            f"MemoryRouter(keyword={type(self._keyword).__name__}, "
            f"vector={type(self._vector).__name__ if self._vector else 'None'}, "
            f"strategy={self._search_strategy})"
        )
