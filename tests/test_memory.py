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
from kairos.gateway.adapters.base import CLIAdapter, TelegramAdapter, SlackAdapter
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


def test_memory_add_get():
    import tempfile, os
    td = tempfile.mkdtemp()
    store = MemoryStore(db_path=os.path.join(td, "test.db"))
    result = store.add("user", "prefers_python", "User likes Python over JS")
    assert result == "added"
    assert store.get("user", "prefers_python") == "User likes Python over JS"



def test_memory_update():
    import tempfile, os
    td = tempfile.mkdtemp()
    store = MemoryStore(db_path=os.path.join(td, "test.db"))
    store.add("memory", "project_dir", "~/old")
    result = store.add("memory", "project_dir", "~/new")
    assert result == "updated"
    assert store.get("memory", "project_dir") == "~/new"



def test_memory_all_and_count():
    import tempfile, os
    td = tempfile.mkdtemp()
    store = MemoryStore(db_path=os.path.join(td, "test.db"))
    store.add("user", "k1", "v1")
    store.add("memory", "k2", "v2")
    assert store.count() == 2
    all_items = store.all()
    assert len(all_items) == 2
    assert store.count("user") == 1



def test_memory_search():
    import tempfile, os
    td = tempfile.mkdtemp()
    store = MemoryStore(db_path=os.path.join(td, "test.db"))
    store.add("memory", "python_version", "Use Python 3.11 for this project")
    results = store.search("project")  # Search by value
    assert len(results) == 1



def test_memory_remove_and_clear():
    import tempfile, os
    td = tempfile.mkdtemp()
    store = MemoryStore(db_path=os.path.join(td, "test.db"))
    store.add("user", "temp", "should be removed")
    assert store.remove("user", "temp") is True
    assert store.get("user", "temp") is None
    assert store.remove("user", "nonexistent") is False



def test_memory_format_for_prompt():
    import tempfile, os
    td = tempfile.mkdtemp()
    store = MemoryStore(db_path=os.path.join(td, "test.db"))
    store.add("user", "prefers_concise", "Yes")
    store.add("memory", "project_root", "~/kairos")
    prompt = store.format_for_prompt()
    assert "prefers_concise" in prompt
    assert "project_root" in prompt



def test_memory_middleware_injection():
    import tempfile, os
    td = tempfile.mkdtemp()
    store = MemoryStore(db_path=os.path.join(td, "test.db"))
    store.add("user", "name", "buer103")
    mw = MemoryMiddleware(memory_store=store)
    state = ThreadState()
    state.messages = [{"role": "system", "content": "You are helpful."}]
    mw.before_agent(state, {})
    assert "buer103" in state.messages[0]["content"]
    assert "MEMORY" in state.messages[0]["content"]


# ── Phase 2: Skill Manager ─────────────────────────────────────

from kairos.skills.manager import SkillManager, SkillStatus



