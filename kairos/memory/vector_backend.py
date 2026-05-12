"""VectorMemoryBackend — ChromaDB-based semantic search backend.

Drop-in replacement for SQLiteBackend with semantic (embedding-based) search.
Requires ``pip install chromadb`` (optional dependency).

Usage::

    from kairos.memory import VectorMemoryBackend

    backend = VectorMemoryBackend()
    backend.save("pref_lang", "User prefers Chinese", category="preference")
    results = backend.search("language preference")  # semantic, not keyword
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

from kairos.memory.backends import MemoryBackend

logger = logging.getLogger("kairos.memory.vector")
_IS_AVAILABLE: bool | None = None  # lazy-detected


def _check_chromadb() -> bool:
    """Lazy import check — only fails when actually used."""
    global _IS_AVAILABLE
    if _IS_AVAILABLE is None:
        try:
            import chromadb  # noqa: F401
            _IS_AVAILABLE = True
        except ImportError:
            _IS_AVAILABLE = False
    return _IS_AVAILABLE


class VectorMemoryBackend(MemoryBackend):
    """Semantic memory backend powered by ChromaDB.

    Stores each memory entry as a ChromaDB document with embeddings.
    ``search()`` uses cosine similarity instead of FTS5 keyword matching.

    Config:
        persist_dir: str — ChromaDB persistence directory (default: ~/.kairos/memory/chroma)
        collection_name: str — collection name (default: "kairos_memory")
        embedding_fn: callable | None — custom embedding function.
                      Default: ChromaDB's built-in all-MiniLM-L6-v2.
        similarity_threshold: float — minimum cosine similarity (default: 0.3)
    """

    def __init__(
        self,
        persist_dir: str | Path | None = None,
        collection_name: str = "kairos_memory",
        embedding_fn: Any = None,
        similarity_threshold: float = 0.3,
    ):
        if not _check_chromadb():
            raise ImportError(
                "chromadb is required for VectorMemoryBackend. "
                "Install with: pip install kairos[chroma] or pip install chromadb"
            )
        import chromadb

        self._persist_dir = str(persist_dir or Path.home() / ".kairos" / "memory" / "chroma")
        self._collection_name = collection_name
        self._similarity_threshold = similarity_threshold
        self._lock = threading.Lock()

        Path(self._persist_dir).mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(path=self._persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

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
        meta = dict(metadata or {})
        meta["category"] = category
        meta["updated_at"] = time.time()
        if ttl is not None:
            meta["expires_at"] = time.time() + ttl

        with self._lock:
            # ChromaDB upsert: delete existing, add new
            existing = self._collection.get(ids=[key])
            if existing and existing["ids"]:
                self._collection.delete(ids=[key])
            self._collection.add(
                ids=[key],
                documents=[value],
                metadatas=[meta],
            )

    def load(self, key: str) -> dict[str, Any] | None:
        self._prune_expired()
        with self._lock:
            result = self._collection.get(ids=[key])
        if not result or not result["ids"]:
            return None
        return self._to_entry(result, 0)

    def delete(self, key: str) -> bool:
        with self._lock:
            existing = self._collection.get(ids=[key])
            if not existing or not existing["ids"]:
                return False
            self._collection.delete(ids=[key])
            return True

    def search(
        self, query: str, *, limit: int = 20, category: str | None = None
    ) -> list[dict[str, Any]]:
        self._prune_expired()
        where_filter = None
        if category:
            where_filter = {"category": category}

        with self._lock:
            results = self._collection.query(
                query_texts=[query],
                n_results=limit,
                where=where_filter,
            )

        entries = []
        if results and results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                distance = results.get("distances", [[0]])[0][i] if results.get("distances") else 0
                similarity = 1.0 - distance if distance else 1.0
                if similarity < self._similarity_threshold:
                    continue
                entry = self._to_entry(results, i)
                entry["_similarity"] = round(similarity, 4)
                entries.append(entry)

        # Sort by similarity descending
        entries.sort(key=lambda e: e.get("_similarity", 0), reverse=True)
        return entries[:limit]

    def count(self) -> int:
        with self._lock:
            return self._collection.count()

    def list_keys(
        self, prefix: str = "", *, category: str | None = None
    ) -> list[str]:
        """List all keys, optionally filtered by prefix and/or category."""
        with self._lock:
            where_filter = None
            if category:
                where_filter = {"category": category}
            all_data = self._collection.get(where=where_filter)
        if not all_data or not all_data["ids"]:
            return []
        keys = list(all_data["ids"])
        if prefix:
            keys = [k for k in keys if k.startswith(prefix)]
        return keys

    def clear(self, *, category: str | None = None) -> int:
        """Remove all (or category-filtered) memories. Returns count removed."""
        with self._lock:
            if category:
                # ChromaDB doesn't support category-filtered delete natively,
                # so we get matching IDs first then delete
                matching = self._collection.get(where={"category": category})
                if matching and matching["ids"]:
                    count = len(matching["ids"])
                    self._collection.delete(ids=matching["ids"])
                    return count
                return 0
            else:
                count = self._collection.count()
                self._client.delete_collection(self._collection_name)
                self._collection = self._client.create_collection(
                    name=self._collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
                return count

    # ── Helpers ──────────────────────────────────────────────────────

    def _to_entry(self, result: Any, index: int) -> dict[str, Any]:
        """Convert ChromaDB query result at index to standard MemoryBackend dict."""
        key = result["ids"][0][index]
        value = result["documents"][0][index] if result.get("documents") else ""
        meta = result["metadatas"][0][index] if result.get("metadatas") else {}
        return {
            "key": key,
            "value": value,
            "category": meta.get("category", ""),
            "metadata": {k: v for k, v in (meta or {}).items()
                         if k not in ("category", "updated_at", "expires_at")},
            "updated_at": meta.get("updated_at", 0),
            "expires_at": meta.get("expires_at"),
        }

    def _prune_expired(self) -> None:
        """Remove entries past their TTL."""
        now = time.time()
        with self._lock:
            all_data = self._collection.get()
            if not all_data or not all_data["ids"]:
                return
            expired_ids = []
            for i, mid in enumerate(all_data["ids"]):
                meta = all_data["metadatas"][i] if all_data.get("metadatas") else {}
                expires = meta.get("expires_at")
                if expires and expires < now:
                    expired_ids.append(mid)
            if expired_ids:
                self._collection.delete(ids=expired_ids)
                logger.debug("Pruned %d expired entries from vector memory", len(expired_ids))

    def __repr__(self) -> str:
        return (
            f"VectorMemoryBackend(persist={self._persist_dir!r}, "
            f"collection={self._collection_name!r}, "
            f"count={self.count()})"
        )
