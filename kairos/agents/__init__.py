"""Agent subsystem — sub-agents, delegation, and factory patterns."""

from kairos.agents.types import SubAgentType, BUILTIN_TYPES, GENERAL_PURPOSE, BASH, RESEARCH
from kairos.agents.factory import register_subagent_types, get_subagent_type, set_executor
from kairos.agents.executor import SubAgentExecutor
from kairos.agents.delegate import (
    DelegateTask,
    DelegateResult,
    DelegateConfig,
    DelegationManager,
    SubAgent,
    register_delegate_tool,
)

__all__ = [
    "SubAgentType",
    "BUILTIN_TYPES",
    "GENERAL_PURPOSE",
    "BASH",
    "RESEARCH",
    "register_subagent_types",
    "get_subagent_type",
    "set_executor",
    "SubAgentExecutor",
    "DelegateTask",
    "DelegateResult",
    "DelegateConfig",
    "DelegationManager",
    "SubAgent",
    "register_delegate_tool",
]
