"""Built-in knowledge lookup tool."""

from kairos.tools.registry import register_tool


@register_tool(
    name="knowledge_lookup",
    description="Query the structured knowledge store by schema and filters.",
    parameters={
        "schema": {"type": "string", "description": "Knowledge schema name to query"},
        "query": {"type": "string", "description": "Natural language query"},
        "filters": {
            "type": "object",
            "description": "Optional key-value filters (e.g. {\"root_cause\": \"overheat\"})",
        },
    },
)
def knowledge_lookup(schema: str, query: str, filters: dict | None = None) -> dict:
    """
    Query structured knowledge. Currently a stub — real implementation
    in kairos.infra.knowledge with user-defined schemas.
    """
    return {
        "schema": schema,
        "query": query,
        "filters": filters,
        "results": [],
        "message": "Knowledge store not configured. Define schemas via KnowledgeSchema base class.",
    }
