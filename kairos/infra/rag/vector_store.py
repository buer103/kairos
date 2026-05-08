"""Vector store abstraction with pluggable backends."""

from __future__ import annotations

from typing import Any


class VectorStore:
    """
    Abstract vector store for RAG retrieval.

    Default: in-memory (no dependencies).
    Production: ChromaDB, FAISS, or any backend implementing the interface.
    """

    def __init__(self, backend: str = "memory"):
        self._backend = backend
        self._documents: list[dict[str, Any]] = []

    def add(self, documents: list[str], metadatas: list[dict] | None = None, ids: list[str] | None = None) -> None:
        """Add documents to the store."""
        for i, doc in enumerate(documents):
            self._documents.append({
                "id": ids[i] if ids else str(len(self._documents)),
                "content": doc,
                "metadata": metadatas[i] if metadatas else {},
            })

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Search for documents similar to the query.

        In-memory backend uses simple keyword matching.
        Production backends use embedding-based similarity.
        """
        # Keyword overlap scoring with partial matching
        query_lower = query.lower()
        scored = []
        for doc in self._documents:
            content_lower = doc["content"].lower()
            # Exact word overlap
            query_terms = set(query_lower.split())
            content_terms = set(content_lower.split())
            exact_overlap = len(query_terms & content_terms)
            # Partial substring matching
            partial = sum(1 for t in query_terms if any(t in ct or ct in t for ct in content_terms))
            score = exact_overlap * 2 + partial
            if score > 0:
                scored.append((score / max(len(query_terms) * 3, 1), doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {"id": d["id"], "content": d["content"], "metadata": d["metadata"], "score": s}
            for s, d in scored[:top_k]
        ]

    def clear(self) -> None:
        """Remove all documents."""
        self._documents = []

    def count(self) -> int:
        """Number of documents in the store."""
        return len(self._documents)
