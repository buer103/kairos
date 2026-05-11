"""Self-registering tool registry with timeout, categories, parallel execution, and smart dispatch.

Features:
  - Self-registering decorator with OpenAI-compatible schema
  - Categories: read_only, write, interactive for safe parallelization
  - Smart dispatch: auto-detects parallel-safe batches, falls back to serial
  - Path-scoped parallelism: file tools with non-overlapping paths run concurrently
  - Timeout enforcement per-tool
  - Tool statistics (calls, errors, duration)
  - Plugin tool registration
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import time
from pathlib import Path
from threading import Lock
from typing import Any, Callable

logger = logging.getLogger("kairos.tools")

_registry: dict[str, dict[str, Any]] = {}
_lock = Lock()
DEFAULT_TIMEOUT = 30  # seconds

# ══════════════════════════════════════════════════════════════
# Parallel execution configuration
# ══════════════════════════════════════════════════════════════

# Tools that must never run concurrently (interactive / user-facing).
_NEVER_PARALLEL_TOOLS: frozenset[str] = frozenset({"clarify"})

# Read-only tools — no shared mutable state, always safe to parallelize.
_PARALLEL_SAFE_TOOLS: frozenset[str] = frozenset({
    "read_file",
    "search_files",
    "session_search",
    "skill_view",
    "skills_list",
    "vision_analyze",
    "web_search",
    "rag_search",
    "knowledge_lookup",
    "list_files",
})

# File tools that can run concurrently when targeting independent paths.
_PATH_SCOPED_TOOLS: frozenset[str] = frozenset({
    "read_file", "write_file", "patch",
})

# Maximum number of concurrent worker threads.
_MAX_TOOL_WORKERS: int = 8


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


# ══════════════════════════════════════════════════════════════
# Smart parallel dispatch
# ══════════════════════════════════════════════════════════════


def _should_parallelize_tool_batch(tool_calls: list[dict]) -> bool:
    """Return True when a tool-call batch is safe to run concurrently.

    A batch is parallel-safe when:
      1. All tools are either _PARALLEL_SAFE_TOOLS or _PATH_SCOPED_TOOLS
      2. No tool is in _NEVER_PARALLEL_TOOLS
      3. Path-scoped tools target independent (non-overlapping) paths
      4. Batch has 2+ calls
    """
    if len(tool_calls) <= 1:
        return False

    tool_names = [tc.get("name", tc.get("function", {}).get("name", "")) for tc in tool_calls]
    if any(name in _NEVER_PARALLEL_TOOLS for name in tool_names):
        return False

    reserved_paths: list[Path] = []
    for tc in tool_calls:
        name = tc.get("name", tc.get("function", {}).get("name", ""))
        raw_args = tc.get("arguments", tc.get("args", {}))
        if isinstance(raw_args, str):
            try:
                args = json.loads(raw_args)
            except Exception:
                return False
        else:
            args = raw_args
        if not isinstance(args, dict):
            return False

        if name in _PATH_SCOPED_TOOLS:
            scoped_path = _extract_parallel_scope_path(name, args)
            if scoped_path is None:
                return False
            if any(_paths_overlap(scoped_path, existing) for existing in reserved_paths):
                return False
            reserved_paths.append(scoped_path)
            continue

        if name not in _PARALLEL_SAFE_TOOLS:
            return False

    return True


def _extract_parallel_scope_path(tool_name: str, args: dict) -> Path | None:
    """Return the normalized file target for path-scoped tools."""
    raw_path = args.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None

    expanded = Path(raw_path).expanduser()
    if expanded.is_absolute():
        return Path(os.path.abspath(str(expanded)))

    return Path(os.path.abspath(str(Path.cwd() / expanded)))


def _paths_overlap(left: Path, right: Path) -> bool:
    """Return True when two paths may refer to the same subtree."""
    left_parts = left.parts
    right_parts = right.parts
    if not left_parts or not right_parts:
        return bool(left_parts) == bool(right_parts) and bool(left_parts)
    common_len = min(len(left_parts), len(right_parts))
    return left_parts[:common_len] == right_parts[:common_len]


def execute_tools_smart(
    tool_calls: list[dict],
    timeout: float | None = None,
    max_workers: int | None = None,
) -> list[dict[str, Any]]:
    """Execute tool calls — parallel when safe, serial otherwise.

    Auto-detects parallel-safe batches using _should_parallelize_tool_batch.
    Falls back to serial `execute_tool` for unsafe batches.

    Args:
        tool_calls: List of tool call dicts with 'name'/'arguments' or
                    'function'/'name'/'arguments' keys.
        timeout: Max seconds per tool (None = use per-tool defaults).
        max_workers: Max concurrent threads (default: _MAX_TOOL_WORKERS).

    Returns:
        List of result dicts in the same order as tool_calls.
    """
    if not tool_calls:
        return []

    # Normalize tool call format
    normalized_calls = []
    for tc in tool_calls:
        func = tc.get("function", tc)
        name = func.get("name", tc.get("name", ""))
        raw_args = func.get("arguments", tc.get("arguments", {}))
        if isinstance(raw_args, str):
            try:
                args = json.loads(raw_args)
            except Exception:
                args = {}
        else:
            args = raw_args
        normalized_calls.append((name, args))

    # Try parallel execution
    if _should_parallelize_tool_batch(tool_calls):
        workers = max_workers or _MAX_TOOL_WORKERS
        logger.debug(
            "Parallel execution: %d tools with %d workers",
            len(normalized_calls), workers,
        )
        return execute_tools_parallel(normalized_calls, timeout=timeout, max_workers=workers)

    # Serial fallback
    results = []
    for name, args in normalized_calls:
        results.append(execute_tool(name, args, timeout=timeout))
    return results
