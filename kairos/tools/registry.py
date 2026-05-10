"""Self-registering tool registry with timeout, categories, and parallel execution."""

from __future__ import annotations

import concurrent.futures
import json
import logging
import time
from threading import Lock
from typing import Any, Callable

logger = logging.getLogger("kairos.tools")

_registry: dict[str, dict[str, Any]] = {}
_lock = Lock()
DEFAULT_TIMEOUT = 30  # seconds


def register_tool(
    name: str,
    description: str,
    parameters: dict[str, Any],
    timeout: float = DEFAULT_TIMEOUT,
    category: str = "general",
    enabled: bool = True,
) -> Callable:
    """Decorator that registers a function as a tool with OpenAI-compatible schema.

    Args:
        name: Tool name (must be unique).
        description: Natural language description for the LLM.
        parameters: JSON Schema properties dict.
        timeout: Max execution time in seconds.
        category: Grouping label for tool management.
        enabled: Whether the tool is active.
    """

    def decorator(fn: Callable) -> Callable:
        with _lock:
            _registry[name] = {
                "fn": fn,
                "timeout": timeout,
                "category": category,
                "enabled": enabled,
                "call_count": 0,
                "error_count": 0,
                "total_duration_ms": 0.0,
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


# ── Query ───────────────────────────────────────────────────────

def get_tool(name: str) -> dict[str, Any] | None:
    return _registry.get(name)


def get_all_tools() -> dict[str, dict[str, Any]]:
    return dict(_registry)


def get_tool_schemas(enabled_only: bool = True, categories: list[str] | None = None) -> list[dict]:
    """Get OpenAI-compatible tool schemas, optionally filtered."""
    schemas = []
    for name, t in _registry.items():
        if enabled_only and not t.get("enabled", True):
            continue
        if categories and t.get("category", "general") not in categories:
            continue
        schemas.append(t["schema"])
    return schemas


def list_tools(category: str | None = None) -> list[dict]:
    """List tool metadata for management."""
    tools = []
    for name, t in _registry.items():
        if category and t.get("category") != category:
            continue
        tools.append({
            "name": name,
            "description": t["schema"]["function"]["description"],
            "category": t.get("category", "general"),
            "enabled": t.get("enabled", True),
            "timeout": t.get("timeout", DEFAULT_TIMEOUT),
            "calls": t.get("call_count", 0),
            "errors": t.get("error_count", 0),
        })
    return sorted(tools, key=lambda x: x["name"])


def tool_stats() -> dict[str, Any]:
    """Aggregate tool statistics."""
    total_calls = sum(t.get("call_count", 0) for t in _registry.values())
    total_errors = sum(t.get("error_count", 0) for t in _registry.values())
    total_duration = sum(t.get("total_duration_ms", 0) for t in _registry.values())
    return {
        "total_tools": len(_registry),
        "enabled": sum(1 for t in _registry.values() if t.get("enabled", True)),
        "disabled": sum(1 for t in _registry.values() if not t.get("enabled", True)),
        "total_calls": total_calls,
        "total_errors": total_errors,
        "error_rate": round(total_errors / max(total_calls, 1), 3),
        "avg_duration_ms": round(total_duration / max(total_calls, 1), 1),
        "categories": list(set(t.get("category", "general") for t in _registry.values())),
    }


# ── Management ──────────────────────────────────────────────────

def register_plugin_tool(
    name: str,
    handler: Callable,
    schema: dict[str, Any],
    category: str = "plugin",
    timeout: float = DEFAULT_TIMEOUT,
) -> bool:
    """Register a tool programmatically (for plugins).

    Unlike the @register_tool decorator, this accepts an already-defined
    handler and schema. Returns True if registered, False if name exists.
    """
    if name in _registry:
        return False
    with _lock:
        if name in _registry:  # double-check
            return False
        _registry[name] = {
            "fn": handler,
            "timeout": timeout,
            "category": category,
            "enabled": True,
            "call_count": 0,
            "error_count": 0,
            "total_duration_ms": 0.0,
            "schema": schema,
        }
    return True


def enable_tool(name: str) -> bool:
    t = _registry.get(name)
    if t:
        t["enabled"] = True
        return True
    return False


def disable_tool(name: str) -> bool:
    t = _registry.get(name)
    if t:
        t["enabled"] = False
        return True
    return False


# ── Execution ───────────────────────────────────────────────────

def execute_tool(name: str, args: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
    """Execute a registered tool with timeout enforcement."""
    tool = _registry.get(name)
    if not tool:
        return {"error": f"Unknown tool: {name}"}

    if not tool.get("enabled", True):
        return {"error": f"Tool disabled: {name}"}

    timeout = timeout or tool.get("timeout", DEFAULT_TIMEOUT)
    start = time.time()

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_call_tool, tool["fn"], args)
            result = future.result(timeout=timeout)

        elapsed = (time.time() - start) * 1000
        tool["call_count"] = tool.get("call_count", 0) + 1
        tool["total_duration_ms"] = tool.get("total_duration_ms", 0) + elapsed

        if isinstance(result, dict) and "error" in result:
            tool["error_count"] = tool.get("error_count", 0) + 1

        if isinstance(result, dict):
            return result
        return {"result": result}

    except concurrent.futures.TimeoutError:
        elapsed = (time.time() - start) * 1000
        tool["error_count"] = tool.get("error_count", 0) + 1
        tool["total_duration_ms"] = tool.get("total_duration_ms", 0) + elapsed
        logger.warning("Tool '%s' timed out after %.1fs", name, timeout)
        return {"error": f"Tool timed out after {timeout}s"}

    except Exception as e:
        elapsed = (time.time() - start) * 1000
        tool["error_count"] = tool.get("error_count", 0) + 1
        tool["total_duration_ms"] = tool.get("total_duration_ms", 0) + elapsed
        logger.error("Tool '%s' failed: %s", name, e)
        return {"error": str(e)}


def execute_tools_parallel(
    calls: list[tuple[str, dict[str, Any]]],
    timeout: float | None = None,
    max_workers: int = 5,
) -> list[dict[str, Any]]:
    """Execute multiple tool calls in parallel. Each tuple is (name, args)."""
    results = [None] * len(calls)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for i, (name, args) in enumerate(calls):
            futures[executor.submit(execute_tool, name, args, timeout)] = i

        for future in concurrent.futures.as_completed(futures, timeout=timeout):
            i = futures[future]
            try:
                results[i] = future.result()
            except Exception as e:
                results[i] = {"error": str(e)}

    return results


def _call_tool(fn: Callable, args: dict) -> Any:
    """Internal: call the tool function with args."""
    return fn(**args)
