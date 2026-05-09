"""Built-in knowledge lookup tool — structured query across typed schemas."""

from __future__ import annotations

from typing import Any

from kairos.tools.registry import register_tool

# Module-level store — injected by Agent or middleware at startup
_knowledge_stores: dict[str, Any] = {}


def set_knowledge_store(name: str, store: Any) -> None:
    """Register a knowledge store under a schema name."""
    _knowledge_stores[name] = store


def get_knowledge_store(name: str) -> Any:
    """Get a registered knowledge store by schema name."""
    return _knowledge_stores.get(name)


@register_tool(
    name="knowledge_lookup",
    description="Query the structured knowledge store by schema and filters. Use this to look up known patterns, facts, or domain data (e.g., fault patterns, reference specs, configuration rules).",
    parameters={
        "schema": {"type": "string", "description": "Knowledge schema name to query (e.g., 'FaultDiagnosis', 'VehicleSpecs')"},
        "query": {"type": "string", "description": "Natural language or keyword query for text search"},
        "filters": {
            "type": "object",
            "description": "Optional key-value filters for exact matching (e.g. {\"root_cause\": \"overheat\"})",
        },
    },
)
def knowledge_lookup(schema: str, query: str, filters: dict | None = None) -> dict:
    """Query structured knowledge by schema name."""
    store = _knowledge_stores.get(schema)
    if store is None:
        available = list(_knowledge_stores.keys())
        return {
            "schema": schema,
            "query": query,
            "filters": filters,
            "results": [],
            "total_found": 0,
            "error": f"No knowledge store registered for schema '{schema}'."
            + (f" Available schemas: {available}" if available else " No stores registered."),
        }

    # Apply filters first if provided, then text search
    if filters:
        candidates = store.query(filters)
        if not candidates:
            return {
                "schema": schema,
                "query": query,
                "filters": filters,
                "results": [],
                "total_found": 0,
            }
        # Text search within filtered candidates (brute force)
        query_lower = query.lower()
        results = [c for c in candidates if query_lower in str(c.to_dict()).lower()]
        return {
            "schema": schema,
            "query": query,
            "filters": filters,
            "results": [r.to_dict() for r in results],
            "total_found": len(results),
        }

    # Pure text search across all items
    results = store.search(query)
    return {
        "schema": schema,
        "query": query,
        "filters": filters,
        "results": [r.to_dict() for r in results],
        "total_found": len(results),
    }
