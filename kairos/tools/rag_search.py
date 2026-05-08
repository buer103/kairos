"""Built-in RAG search tool."""

from kairos.tools.registry import register_tool


@register_tool(
    name="rag_search",
    description="Search the knowledge base for relevant information using semantic search.",
    parameters={
        "query": {"type": "string", "description": "Search query"},
        "top_k": {"type": "integer", "description": "Number of results to return (default: 5)"},
    },
)
def rag_search(query: str, top_k: int = 5) -> dict:
    """
    Search the knowledge base. Currently a stub — real implementation
    in kairos.infra.rag with vector store backend.
    """
    return {
        "query": query,
        "results": [],
        "message": "RAG engine not configured. Install with: pip install kairos-agent[rag]",
    }
