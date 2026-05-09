"""Integration tests for Kairos framework."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

import json
import tempfile
from pathlib import Path
import pytest
from kairos.infra.knowledge.schema import KnowledgeSchema
from kairos.infra.knowledge.store import KnowledgeStore
from kairos.infra.rag.vector_store import VectorStore
from kairos.infra.rag.adapters import MarkdownAdapter, TextAdapter
from kairos.core.state import Case, Step, ThreadState, merge_artifacts
from kairos.core.middleware import MiddlewarePipeline
from kairos.middleware import (
    EvidenceTracker,
    ConfidenceScorer,
    ContextCompressor,
    SkillLoader,
    Skill,
)
from kairos.middleware.evidence import EvidenceTracker as ET
from kairos.infra.evidence.tracker import EvidenceDB
from kairos.chat.session import SessionStore
from kairos.tools.rag_search import rag_search, set_rag_store
from kairos.tools.knowledge_lookup import knowledge_lookup, set_knowledge_store
from kairos.tools.registry import get_all_tools, get_tool_schemas, execute_tool
from kairos.prompt.template import PromptBuilder
from kairos.agents.types import SubAgentType, BUILTIN_TYPES, GENERAL_PURPOSE, BASH, RESEARCH
from kairos.agents.factory import register_subagent_types, get_subagent_type
from kairos.memory.store import MemoryStore
from kairos.memory.middleware import MemoryMiddleware
from kairos.skills.manager import SkillManager, SkillStatus
from kairos.session.search import SessionSearch
from kairos.sandbox import (
    LocalSandbox, DockerSandbox, SSHSandbox,
    SandboxConfig, SandboxProvider, create_sandbox,
)
from kairos.gateway.protocol import (
    UnifiedMessage, UnifiedResponse, ContentBlock, ContentType,
    MessageRole, ConnectionState,
)
from kairos.gateway.adapters.base import CLIAdapter
from kairos.gateway.adapters import TelegramAdapter, SlackAdapter
from kairos.training.recorder import TrajectoryRecorder, ToolContext
from kairos.training.env import (
    TrainingEnv, EnvironmentRegistry, RolloutRunner,
    reward_confidence, reward_success_rate, reward_evidence_quality, reward_file_creation,
)
from kairos.middleware.dangling import DanglingToolCallMiddleware
from kairos.middleware.subagent_limit import SubagentLimitMiddleware
from kairos.middleware.clarify import ClarificationMiddleware
from kairos.middleware.thread_data import ThreadDataMiddleware
from kairos.middleware.todo import TodoMiddleware
from kairos.middleware.title import TitleMiddleware
from kairos.middleware.uploads import UploadsMiddleware
from kairos.middleware.view_image import ViewImageMiddleware
from kairos.providers.credential import CredentialPool, Credential, RetryConfig
from kairos.middleware.llm_retry import LLMRetryMiddleware, ToolArgRepairMiddleware
from kairos.core.stateful_agent import StatefulAgent
from kairos.providers.base import ModelConfig


def test_local_sandbox_execute():
    sandbox = LocalSandbox()
    assert sandbox.is_available()
    result = sandbox.execute("echo hello")
    assert result.exit_code == 0
    assert "hello" in result.output



def test_local_sandbox_timeout():
    sandbox = LocalSandbox(config=SandboxConfig(timeout=1))
    result = sandbox.execute("sleep 5")
    assert result.exit_code == -1



def test_sandbox_factory():
    cfg = SandboxConfig(provider=SandboxProvider.LOCAL)
    sandbox = create_sandbox(cfg)
    assert isinstance(sandbox, LocalSandbox)



def test_docker_sandbox_unavailable():
    sandbox = DockerSandbox()
    if not sandbox.is_available():
        result = sandbox.execute("echo hello")
        assert result.exit_code == -1



def test_ssh_sandbox_no_host():
    sandbox = SSHSandbox(config=SandboxConfig(provider=SandboxProvider.SSH))
    assert not sandbox.is_available()
    result = sandbox.execute("echo hello")
    assert result.exit_code == -1


# ── Phase 3: Gateway Protocol ──────────────────────────────────

from kairos.gateway.protocol import (
    UnifiedMessage, UnifiedResponse, ContentBlock, ContentType,
    MessageRole, ConnectionState,
)
from kairos.gateway.adapters.base import CLIAdapter
from kairos.gateway.adapters import TelegramAdapter, SlackAdapter



class TestSandboxMiddleware:
    def test_passthrough_no_sandbox(self):
        from kairos.middleware import SandboxMiddleware
        mw = SandboxMiddleware()
        assert not mw.is_active
        # Should passthrough
        called = []
        def original(name, args):
            called.append(name)
            return {"stdout": "ok"}
        result = mw.wrap_tool_call("terminal", {"command": "echo hi"}, original)
        assert result["stdout"] == "ok"
        assert called == ["terminal"]

    def test_local_sandbox_intercept(self):
        from kairos.middleware import SandboxMiddleware
        from kairos.sandbox.providers import SandboxConfig, SandboxProvider
        config = SandboxConfig(provider=SandboxProvider.LOCAL)
        mw = SandboxMiddleware(sandbox_config=config)
        # Local sandbox wraps terminal calls
        result = mw.wrap_tool_call(
            "terminal",
            {"command": "echo hello"},
            lambda n, a: {"stdout": "passthrough"},
        )
        assert "hello" in result.get("stdout", "")
        assert result.get("exit_code") == 0

    def test_non_terminal_passthrough(self):
        from kairos.middleware import SandboxMiddleware
        from kairos.sandbox.providers import SandboxConfig, SandboxProvider
        config = SandboxConfig(provider=SandboxProvider.LOCAL)
        mw = SandboxMiddleware(sandbox_config=config)
        called = []
        result = mw.wrap_tool_call(
            "read_file",
            {"path": "/tmp/x"},
            lambda n, a: called.append(n) or {"ok": True},
        )
        assert called == ["read_file"]



