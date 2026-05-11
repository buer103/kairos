"""Sub-agent tool policy — whitelist/blacklist for delegated agents (DeerFlow parity).

Controls which tools sub-agents can access, preventing privilege escalation
and reducing attack surface for delegated tasks.

Usage:
    policy = SubAgentPolicy(
        whitelist=["read_file", "search_files", "terminal"],
        blacklist=["gateway_start", "cron_add"],
        max_iterations=10,
    )
    sub_agent = SubAgentFactory.create(
        parent_agent, policy=policy,
    )
"""

from __future__ import annotations

from typing import Any


class SubAgentPolicy:
    """Access control for sub-agent tool usage.

    DeerFlow-compatible: restricts delegated agents to a safe subset
    of the parent agent's tools.

    Priority: blacklist > whitelist > default_allow
    """

    def __init__(
        self,
        whitelist: list[str] | None = None,
        blacklist: list[str] | None = None,
        max_iterations: int = 10,
        max_tokens: int = 32000,
        allow_delegation: bool = False,
        default_allow: bool = False,
    ):
        """Initialize sub-agent policy.

        Args:
            whitelist: Only these tools are allowed. Empty = all allowed
                       (unless blacklist is set).
            blacklist: These tools are forbidden regardless of whitelist.
            max_iterations: Max tool-calling iterations for sub-agent.
            max_tokens: Max token budget for sub-agent.
            allow_delegation: Allow sub-agent to spawn its own sub-agents.
            default_allow: When True, allow unknown tools. When False,
                          only whitelist or explicitly-allowed tools pass.
        """
        self.whitelist: set[str] = set(whitelist or [])
        self.blacklist: set[str] = set(blacklist or [])
        self.max_iterations: int = max_iterations
        self.max_tokens: int = max_tokens
        self.allow_delegation: bool = allow_delegation
        self.default_allow: bool = default_allow

    def is_allowed(self, tool_name: str) -> bool:
        """Check if a tool is allowed for this sub-agent."""
        # Blacklist takes priority
        if self.blacklist and tool_name in self.blacklist:
            return False
        # Whitelist check
        if self.whitelist:
            return tool_name in self.whitelist
        # Default
        return self.default_allow

    def filter_tools(self, tool_schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Filter OpenAI tool schemas to only allowed tools."""
        return [
            s for s in tool_schemas
            if self.is_allowed(s.get("function", {}).get("name", ""))
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "whitelist": sorted(self.whitelist),
            "blacklist": sorted(self.blacklist),
            "max_iterations": self.max_iterations,
            "max_tokens": self.max_tokens,
            "allow_delegation": self.allow_delegation,
            "default_allow": self.default_allow,
        }

    @classmethod
    def read_only(cls, max_iterations: int = 10) -> "SubAgentPolicy":
        """Pre-configured: read-only tools only (safe for code review)."""
        return cls(
            whitelist=[
                "read_file", "search_files", "list_files",
                "skill_view", "skills_list", "session_search",
                "web_search", "rag_search",
            ],
            max_iterations=max_iterations,
            allow_delegation=False,
        )

    @classmethod
    def code_assistant(cls, max_iterations: int = 20) -> "SubAgentPolicy":
        """Pre-configured: coding tools (read + terminal + write)."""
        return cls(
            whitelist=[
                "read_file", "search_files", "list_files",
                "terminal", "write_file", "patch",
                "skill_view", "skills_list",
            ],
            max_iterations=max_iterations,
            allow_delegation=False,
        )

    @classmethod
    def full_access(cls, max_iterations: int = 30) -> "SubAgentPolicy":
        """Full access with dangerous tools blacklisted."""
        return cls(
            blacklist=[
                "gateway_start", "gateway_stop",
                "cron_add", "cron_remove",
            ],
            max_iterations=max_iterations,
            default_allow=True,
            allow_delegation=True,
        )


class SubAgentPolicyRegistry:
    """Named policy presets for quick access."""

    _presets: dict[str, SubAgentPolicy] = {}

    @classmethod
    def register(cls, name: str, policy: SubAgentPolicy) -> None:
        cls._presets[name] = policy

    @classmethod
    def get(cls, name: str) -> SubAgentPolicy | None:
        return cls._presets.get(name)

    @classmethod
    def list_presets(cls) -> list[str]:
        return sorted(cls._presets.keys())


# Register default presets
SubAgentPolicyRegistry.register("read-only", SubAgentPolicy.read_only())
SubAgentPolicyRegistry.register("code-assistant", SubAgentPolicy.code_assistant())
SubAgentPolicyRegistry.register("full-access", SubAgentPolicy.full_access())
