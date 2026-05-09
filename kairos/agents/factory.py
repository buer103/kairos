"""Sub-Agent factory — create and execute typed sub-agents.

Provides:
  - register_subagent_types() — register custom sub-agent types
  - get_subagent_type() — lookup by name
  - task() — the tool that Agent calls to delegate work
  - SubAgentExecutor — programmatic execution
"""

from __future__ import annotations

from typing import Any

from kairos.agents.executor import SubAgentExecutor
from kairos.agents.types import BUILTIN_TYPES, SubAgentType
from kairos.tools.registry import register_tool

# Module-level executor — set by Agent during initialization
_executor: SubAgentExecutor | None = None


def set_executor(executor: SubAgentExecutor) -> None:
    """Set the global sub-agent executor (called by Agent.__init__)."""
    global _executor
    _executor = executor


def get_executor() -> SubAgentExecutor | None:
    """Get the current sub-agent executor."""
    return _executor


def register_subagent_types(types: dict[str, SubAgentType]) -> None:
    """Register custom sub-agent types. Merges with built-ins."""
    BUILTIN_TYPES.update(types)


def get_subagent_type(name: str) -> SubAgentType | None:
    """Get a sub-agent type definition by name."""
    return BUILTIN_TYPES.get(name)


@register_tool(
    name="task",
    description="Delegate a task to a typed sub-agent for parallel or isolated execution. Sub-agents have their own context and tool access.",
    parameters={
        "description": {"type": "string", "description": "3-5 word summary of the task"},
        "prompt": {"type": "string", "description": "Detailed task description for the sub-agent"},
        "subagent_type": {
            "type": "string",
            "description": "Type of sub-agent: general-purpose, bash, research, or custom",
        },
        "max_turns": {
            "type": "integer",
            "description": "Override default max turns (optional)",
        },
    },
)
def task(
    description: str,
    prompt: str,
    subagent_type: str = "general-purpose",
    max_turns: int | None = None,
) -> dict[str, Any]:
    """Delegate a task to a sub-agent."""
    if _executor is None:
        return {
            "error": "Sub-agent executor not initialized. The Agent must be created with sub-agent support.",
            "subagent_type": subagent_type,
            "description": description,
        }

    sub_type = BUILTIN_TYPES.get(subagent_type)
    if not sub_type:
        return {
            "error": f"Unknown sub-agent type: {subagent_type}. Available: {list(BUILTIN_TYPES.keys())}",
            "subagent_type": subagent_type,
        }

    # Apply max_turns override
    if max_turns is not None:
        sub_type = SubAgentType(
            name=sub_type.name,
            description=sub_type.description,
            tools=sub_type.tools,
            disallowed_tools=sub_type.disallowed_tools,
            max_turns=max_turns,
            timeout_seconds=sub_type.timeout_seconds,
            model=sub_type.model,
            system_prompt=sub_type.system_prompt,
        )

    result = _executor.run_sync(prompt, sub_type)

    if result.status == "error":
        return {"error": result.error, "subagent_type": subagent_type}

    return {
        "subagent_type": subagent_type,
        "description": description,
        "status": result.status,
        "content": result.content,
        "confidence": result.confidence,
        "evidence": result.evidence,
    }
