"""
Kairos — The right tool, at the right moment.
An AI agent framework inheriting from Hermes and DeerFlow.
"""

__version__ = "0.3.0"

from kairos.core.loop import Agent
from kairos.tools.registry import register_tool

# Register built-in tools
from kairos.tools import rag_search, knowledge_lookup  # noqa: F401
from kairos.agents import factory  # noqa: F401

# Phase 2
from kairos.memory import MemoryStore, MemoryMiddleware
from kairos.skills import SkillManager, SkillStatus, SkillEntry
from kairos.session import SessionSearch
from kairos.sandbox import (
    Sandbox, SandboxConfig, SandboxProvider, SandboxResult,
    LocalSandbox, DockerSandbox, SSHSandbox, create_sandbox,
)

# Phase 3
from kairos.gateway import (
    UnifiedMessage, UnifiedResponse, ContentBlock, ContentType, MessageRole,
    ConnectionState, GatewayServer,
    PlatformAdapter, CLIAdapter, TelegramAdapter, WeChatAdapter, SlackAdapter,
)
from kairos.training import (
    TrajectoryRecorder, ToolContext,
    TrainingEnv, EnvironmentRegistry, RolloutRunner,
    reward_confidence, reward_success_rate, reward_evidence_quality, reward_file_creation,
)

__all__ = [
    "Agent",
    "register_tool",
    "__version__",
    # Phase 2
    "MemoryStore", "MemoryMiddleware",
    "SkillManager", "SkillStatus", "SkillEntry",
    "SessionSearch",
    "Sandbox", "SandboxConfig", "SandboxProvider", "SandboxResult",
    "LocalSandbox", "DockerSandbox", "SSHSandbox", "create_sandbox",
    # Phase 3
    "UnifiedMessage", "UnifiedResponse", "ContentBlock", "ContentType", "MessageRole",
    "ConnectionState", "GatewayServer",
    "PlatformAdapter", "CLIAdapter", "TelegramAdapter", "WeChatAdapter", "SlackAdapter",
    "TrajectoryRecorder", "ToolContext",
    "TrainingEnv", "EnvironmentRegistry", "RolloutRunner",
    "reward_confidence", "reward_success_rate", "reward_evidence_quality", "reward_file_creation",
]
