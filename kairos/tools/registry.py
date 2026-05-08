"""Self-registering tool registry."""

from __future__ import annotations

import json
from typing import Any, Callable


_registry: dict[str, dict[str, Any]] = {}


def register_tool(
    name: str,
    description: str,
    parameters: dict[str, Any],
) -> Callable:
    """Decorator that registers a function as a tool with OpenAI-compatible schema."""

    def decorator(fn: Callable) -> Callable:
        _registry[name] = {
            "fn": fn,
            "schema": {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": {
                        "type": "object",
                        "properties": parameters,
                        "required": list(parameters.keys()),
                    },
                },
            },
        }
        return fn

    return decorator


def get_tool(name: str) -> dict[str, Any] | None:
    """Get a registered tool by name."""
    return _registry.get(name)


def get_all_tools() -> dict[str, dict[str, Any]]:
    """Get all registered tools."""
    return dict(_registry)


def get_tool_schemas() -> list[dict[str, Any]]:
    """Get OpenAI-compatible tool schemas for all registered tools."""
    return [t["schema"] for t in _registry.values()]


def execute_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Execute a registered tool by name with given arguments."""
    tool = _registry.get(name)
    if not tool:
        return {"error": f"Unknown tool: {name}"}
    try:
        result = tool["fn"](**args)
        if isinstance(result, dict):
            return result
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}
