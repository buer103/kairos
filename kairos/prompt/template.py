"""System prompt template engine — DeerFlow-style modular prompting."""

from __future__ import annotations

from typing import Any, Callable

from kairos.prompt.defaults import (
    DEFAULT_GUIDELINES,
    DEFAULT_RESPONSE_STYLE,
    DEFAULT_SOUL,
    DEFAULT_TEMPLATE,
    get_default_template,
)
from kairos.tools.registry import get_all_tools


class PromptBuilder:
    """
    Build system prompts from modular templates.

    Three levels of customization:
      1. Override specific blocks (soul, response_style, guidelines, ...)
      2. Replace the entire template string
      3. Post-render hook for programmatic modification
    """

    def __init__(
        self,
        template: str | None = None,
        mode: str = "default",
        agent_name: str = "Kairos",
        role_description: str = "You are a helpful AI assistant.",
        soul: str | None = None,
        guidelines: str | None = None,
        response_style: str | None = None,
        knowledge_description: str | None = None,
        memory_description: str | None = None,
        post_hook: Callable[[str], str] | None = None,
        **extra_vars,
    ):
        # Template: user-provided > mode-based default
        self._template = template or get_default_template(mode)

        # Block overrides (None = use default)
        self._agent_name = agent_name
        self._role_description = role_description
        self._soul = soul or DEFAULT_SOUL
        self._guidelines = guidelines or DEFAULT_GUIDELINES
        self._response_style = response_style or DEFAULT_RESPONSE_STYLE
        self._knowledge_description = knowledge_description
        self._memory_description = memory_description
        self._post_hook = post_hook
        self._extra_vars = extra_vars

    def build(self) -> str:
        """Render the system prompt with all variables filled."""
        # Auto-generate tools section from registry
        tools = get_all_tools()
        if tools:
            tool_lines = []
            for name, info in tools.items():
                schema = info.get("schema", {})
                func = schema.get("function", {})
                desc = func.get("description", "")
                tool_lines.append(f"  - {name}: {desc}")
            tools_section = "<tools>\n" + "\n".join(tool_lines) + "\n</tools>"
        else:
            tools_section = "<tools>\n  (no tools registered)\n</tools>"

        # Knowledge section
        if self._knowledge_description:
            knowledge_section = f"<knowledge>\n{self._knowledge_description}\n</knowledge>"
        else:
            knowledge_section = ""

        # Memory section
        if self._memory_description:
            memory_section = f"<memory>\n{self._memory_description}\n</memory>"
        else:
            memory_section = ""

        # Personality section
        personality = f"<personality>\n{self._soul}\n</personality>" if self._soul else ""

        # Render
        prompt = self._template.format(
            agent_name=self._agent_name,
            role_description=self._role_description,
            personality=personality,
            tools_section=tools_section,
            knowledge_section=knowledge_section,
            memory_section=memory_section,
            guidelines=self._guidelines,
            response_style=self._response_style,
            **self._extra_vars,
        )

        # Post-hook
        if self._post_hook:
            prompt = self._post_hook(prompt)

        return prompt.strip()

    @classmethod
    def minimal(cls, agent_name: str = "Kairos", **kwargs) -> "PromptBuilder":
        """Quick minimal builder for simple agents."""
        return cls(mode="minimal", agent_name=agent_name, **kwargs)

    @classmethod
    def diagnostic(cls, agent_name: str = "Kairos", **kwargs) -> "PromptBuilder":
        """Builder pre-configured for diagnostic/analysis agents."""
        return cls(mode="diagnostic", agent_name=agent_name, **kwargs)
