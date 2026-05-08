"""Sub-Agent type definitions — configurable agent types for delegation."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SubAgentType:
    """Definition of a sub-agent type.

    Like DeerFlow's general-purpose / bash types, but user-definable.
    """

    name: str
    description: str = ""
    # Tool control: None = inherit all, list = whitelist
    tools: list[str] | None = None
    disallowed_tools: list[str] = field(default_factory=list)
    max_turns: int = 30
    timeout_seconds: int = 900
    model: str = "inherit"
    system_prompt: str = ""


# Built-in types (like DeerFlow's general-purpose and bash)
GENERAL_PURPOSE = SubAgentType(
    name="general-purpose",
    description="Full-capability sub-agent for complex multi-step tasks.",
    disallowed_tools=["task"],  # No recursive delegation
    max_turns=50,
)

BASH = SubAgentType(
    name="bash",
    description="Minimal sub-agent for terminal/file operations only.",
    tools=["read_file", "write_file", "terminal"],  # Whitelist
    disallowed_tools=["task"],
    max_turns=30,
    timeout_seconds=600,
)

RESEARCH = SubAgentType(
    name="research",
    description="Sub-agent specialized in knowledge retrieval and analysis.",
    disallowed_tools=["task", "terminal"],
    max_turns=30,
    system_prompt="You are a research specialist. Search thoroughly and cite sources.",
)

BUILTIN_TYPES: dict[str, SubAgentType] = {
    "general-purpose": GENERAL_PURPOSE,
    "bash": BASH,
    "research": RESEARCH,
}
