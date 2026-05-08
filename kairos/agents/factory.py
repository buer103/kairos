"""Sub-Agent factory — create and execute typed sub-agents."""

from __future__ import annotations

import concurrent.futures
import uuid
from typing import Any

from kairos.agents.types import BUILTIN_TYPES, SubAgentType
from kairos.core.loop import Agent
from kairos.core.state import Case
from kairos.providers.base import ModelConfig, ModelProvider
from kairos.tools.registry import execute_tool, get_tool_schemas, register_tool

# Thread pool for parallel sub-agent execution
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)


def register_subagent_types(types: dict[str, SubAgentType]) -> None:
    """Register custom sub-agent types. Merges with built-ins."""
    BUILTIN_TYPES.update(types)


def get_subagent_type(name: str) -> SubAgentType | None:
    """Get a sub-agent type definition by name."""
    return BUILTIN_TYPES.get(name)


def _run_subagent(
    task_prompt: str,
    sub_type: SubAgentType,
    model: ModelProvider,
    parent_case: Case | None = None,
) -> dict[str, Any]:
    """Execute a single sub-agent and return its result."""
    # Build a minimal agent for this task
    agent = Agent(
        model=ModelConfig(
            api_key=model.config.api_key,
            base_url=model.config.base_url,
            model=model.config.model,
        ),
        middlewares=[],
        agent_name=f"SubAgent({sub_type.name})",
        role_description=sub_type.system_prompt or f"Sub-agent of type: {sub_type.name}",
        max_iterations=sub_type.max_turns,
    )

    # Sub-agent gets its own case, linked to parent
    sub_case = Case(id=f"sub_{uuid.uuid4().hex[:6]}")
    sub_case.conclusion = None

    result = agent.run(task_prompt)

    # Merge sub-agent evidence into parent case
    if parent_case:
        for step in sub_case.steps:
            parent_case.steps.append(step)
        if sub_case.conclusion:
            parent_case.conclusion = sub_case.conclusion

    return result


@register_tool(
    name="task",
    description="Delegate a task to a typed sub-agent for parallel or isolated execution.",
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
    """
    Delegate a task to a sub-agent.

    Currently runs synchronously. Parallel batch execution coming in Phase 2.
    """
    sub_type = get_subagent_type(subagent_type)
    if not sub_type:
        return {"error": f"Unknown sub-agent type: {subagent_type}. Available: {list(BUILTIN_TYPES.keys())}"}

    if max_turns:
        sub_type.max_turns = max_turns

    # Run sub-agent (model is resolved from current agent context in real usage)
    # For now, stub the execution
    return {
        "subagent_type": subagent_type,
        "description": description,
        "prompt": prompt,
        "status": "stub — sub-agent execution requires parent agent context",
        "result": None,
    }
