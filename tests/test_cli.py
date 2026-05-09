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


class TestKairosConsole:
    def test_init(self):
        from kairos.cli import KairosConsole
        c = KairosConsole(skin="default")
        assert c.skin_name == "default"
        assert not c.verbose

    def test_set_skin(self):
        from kairos.cli import KairosConsole
        c = KairosConsole(skin="default")
        assert c.set_skin("hacker")
        assert c.skin_name == "hacker"
        assert not c.set_skin("nonexistent")

    def test_history_tracks(self):
        from kairos.cli import KairosConsole
        c = KairosConsole(verbose=False)
        c.user_input("hello")
        c.agent_output("hi there", confidence=0.95)
        assert len(c._history) == 2
        assert c._history[0]["role"] == "user"
        assert c._history[1]["role"] == "agent"
        assert c._history[1]["confidence"] == 0.95

    def test_verbose_tool_call(self, capsys):
        from kairos.cli import KairosConsole
        c = KairosConsole(verbose=True)
        c.tool_call("read_file", {"path": "/tmp/x"}, {"content": "abc"}, duration_ms=42)
        captured = capsys.readouterr()
        assert "read_file" in captured.out
        assert "42ms" in captured.out



