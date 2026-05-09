"""
Kairos — The right tool, at the right moment.
An AI agent framework inheriting from Hermes and DeerFlow.
"""

__version__ = "0.10.0"

from kairos.core.loop import Agent
from kairos.core.stateful_agent import StatefulAgent
from kairos.core.paths import ThreadPaths
from kairos.core.thread_state import ThreadDataState
from kairos.tools.registry import register_tool

# Register built-in tools (imports trigger @register_tool decorators)
from kairos.tools import rag_search, knowledge_lookup  # noqa: F401
from kairos.tools import builtin  # noqa: F401 — read_file, write_file, terminal, etc.
from kairos.tools import skills_tool  # noqa: F401 — skills_list, skill_view, skill_manage
from kairos.agents import factory  # noqa: F401

# Logging
from kairos.logging import get_logger, log_agent_event, log_tool_call, log_error

# Plugin system
from kairos.plugins import PluginManager, PluginManifest

# Credential
from kairos.providers.credential import CredentialPool, Credential, RetryConfig

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
    DiscordAdapter, FeishuAdapter, WhatsAppAdapter, SignalAdapter,
    LineAdapter, MatrixAdapter, IRCAdapter,
)
from kairos.training import (
    TrajectoryRecorder, ToolContext,
    TrainingEnv, EnvironmentRegistry, RolloutRunner,
    reward_confidence, reward_success_rate, reward_evidence_quality, reward_file_creation,
)

# Phase 4 — Cron, Rich TUI, Sandbox middleware, Delegation
from kairos.cron import CronScheduler, Job, CronSchedule, JobStatus
from kairos.cli import KairosConsole, SKINS
from kairos.middleware.sandbox_mw import SandboxMiddleware
from kairos.agents.delegate import (
    DelegateTask, DelegateResult, DelegateConfig,
    DelegationManager, SubAgent, register_delegate_tool,
)

# Config system
from kairos.config import Config, get_config, write_default_config

__all__ = [
    "Agent", "StatefulAgent",
    "register_tool", "__version__",
    "get_logger", "log_agent_event", "log_tool_call", "log_error",
    "PluginManager", "PluginManifest",
    "CredentialPool", "Credential", "RetryConfig",
    "MemoryStore", "MemoryMiddleware",
    "SkillManager", "SkillStatus", "SkillEntry",
    "SessionSearch",
    "Sandbox", "SandboxConfig", "SandboxProvider", "SandboxResult",
    "LocalSandbox", "DockerSandbox", "SSHSandbox", "create_sandbox",
    "UnifiedMessage", "UnifiedResponse", "ContentBlock", "ContentType", "MessageRole",
    "ConnectionState", "GatewayServer",
    "PlatformAdapter", "CLIAdapter", "TelegramAdapter", "WeChatAdapter", "SlackAdapter",
    "DiscordAdapter", "FeishuAdapter", "WhatsAppAdapter", "SignalAdapter",
    "LineAdapter", "MatrixAdapter", "IRCAdapter",
    "TrajectoryRecorder", "ToolContext",
    "TrainingEnv", "EnvironmentRegistry", "RolloutRunner",
    "reward_confidence", "reward_success_rate", "reward_evidence_quality", "reward_file_creation",
    "CronScheduler", "Job", "CronSchedule", "JobStatus",
    "KairosConsole", "SKINS",
    "SandboxMiddleware",
    "DelegateTask", "DelegateResult", "DelegateConfig",
    "DelegationManager", "SubAgent", "register_delegate_tool",
    "Config", "get_config", "write_default_config",
]
