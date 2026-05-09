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


def test_session_save_load(tmp_path):
    store = SessionStore(base_path=tmp_path)
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]
    store.save("session-1", messages)
    loaded = store.load("session-1")
    assert loaded is not None
    assert len(loaded["messages"]) == 3



def test_session_list(tmp_path):
    store = SessionStore(base_path=tmp_path)
    store.save("s1", [{"role": "user", "content": "hi"}])
    store.save("s2", [{"role": "user", "content": "hey"}])
    sessions = store.list_sessions()
    assert len(sessions) == 2



def test_session_delete(tmp_path):
    store = SessionStore(base_path=tmp_path)
    store.save("s1", [{"role": "user", "content": "hi"}])
    assert store.delete("s1") is True
    assert store.load("s1") is None
    assert store.delete("nonexistent") is False


# ── Evidence DB ─────────────────────────────────────────────────


def test_session_search_index_and_search(tmp_path):
    ss = SessionSearch(db_path=tmp_path / "sessions.db")
    ss.index_session("s1", [
        {"role": "user", "content": "How to fix segmentation fault?"},
        {"role": "assistant", "content": "Check for null pointer dereference first."},
    ])
    results = ss.search("segmentation fault")
    assert len(results) > 0
    assert results[0]["session_id"] == "s1"



def test_session_search_multiple_sessions(tmp_path):
    ss = SessionSearch(db_path=tmp_path / "sessions.db")
    ss.index_session("s1", [{"role": "user", "content": "Python question"}])
    ss.index_session("s2", [{"role": "user", "content": "Rust question"}])
    assert len(ss.search("Python")) == 1
    assert len(ss.search("Rust")) == 1
    assert ss.count() == 2



def test_session_search_recent(tmp_path):
    ss = SessionSearch(db_path=tmp_path / "sessions.db")
    ss.index_session("s1", [{"role": "user", "content": "Hello"}])
    recent = ss.recent_sessions()
    assert len(recent) == 1



def test_session_search_delete(tmp_path):
    ss = SessionSearch(db_path=tmp_path / "sessions.db")
    ss.index_session("s1", [{"role": "user", "content": "Hi"}])
    assert ss.delete_session("s1") is True
    assert ss.count() == 0


# ── Phase 2: Sandbox ────────────────────────────────────────────

from kairos.sandbox import (
    LocalSandbox, DockerSandbox, SSHSandbox,
    SandboxConfig, SandboxProvider, create_sandbox,
)



