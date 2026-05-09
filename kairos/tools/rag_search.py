"""Built-in RAG search tool — semantic search across knowledge bases."""

from __future__ import annotations

from typing import Any

from kairos.tools.registry import register_tool

# Module-level store — injected by Agent or middleware at startup
_vector_store: Any = None


def set_rag_store(store: Any) -> None:
    """Inject a VectorStore instance for RAG searches."""
    global _vector_store
    _vector_store = store


def get_rag_store() -> Any:
    """Get the current RAG store."""
    return _vector_store


@register_tool(
    name="rag_search",
    description="Search the knowledge base for relevant information using semantic search. Use this when you need to find documents, patterns, or reference material related to a topic.",
    parameters={
        "query": {"type": "string", "description": "Search query — be specific and include key terms"},
        "top_k": {"type": "integer", "description": "Number of results to return (default: 5, max: 20)"},
    },
)
def rag_search(query: str, top_k: int = 5) -> dict:
    """Search the RAG index for documents matching the query."""
    if _vector_store is None:
        return {
            "query": query,
            "results": [],
            "error": "RAG store not initialized. Load documents via VectorStore.add() first.",
        }

    top_k = max(1, min(top_k, 20))
    results = _vector_store.search(query, top_k)
    return {
        "query": query,
        "results": [
            {
                "id": r["id"],
                "content": r["content"],
                "metadata": r.get("metadata", {}),
                "score": round(r.get("score", 0), 4),
            }
            for r in results
        ],
        "total_found": len(results),
    }
