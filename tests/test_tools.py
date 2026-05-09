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


def test_all_tools_registered():
    """Verify all 3 built-in tools are auto-registered."""
    tools = get_all_tools()
    assert "rag_search" in tools
    assert "knowledge_lookup" in tools
    assert "task" in tools



def test_tool_schemas():
    schemas = get_tool_schemas()
    assert len(schemas) >= 9  # 3 built-in + 6 new tools
    for s in schemas:
        assert s["type"] == "function"
        assert "name" in s["function"]



def test_execute_unknown_tool():
    result = execute_tool("nonexistent", {})
    assert "error" in result


# ── RAG Search ──────────────────────────────────────────────────


class TestRAGSearch:
    def test_uninitialized_store(self):
        # Reset store for clean test
        from kairos.tools.rag_search import set_rag_store
        set_rag_store(None)
        result = rag_search("test")
        assert "error" in result
        assert "not initialized" in result["error"]

    def test_search_with_store(self):
        store = VectorStore(backend="memory")
        store.add(
            ["Python is a programming language", "JavaScript runs in browsers"],
            metadatas=[{"lang": "python"}, {"lang": "js"}],
        )
        set_rag_store(store)
        result = rag_search("Python programming", top_k=2)
        assert "results" in result
        # Should find at least the Python document
        assert any("Python" in r["content"] for r in result["results"])


# ── Knowledge Lookup ───────────────────────────────────────────

class FaultPattern(KnowledgeSchema):
    def __init__(self, id: str, signal_name: str, root_cause: str, solution: str, confidence: float = 0.0):
        super().__init__(id=id)
        self.signal_name = signal_name
        self.root_cause = root_cause
        self.solution = solution
        self.confidence = confidence



class TestKnowledgeLookup:
    def test_unregistered_schema(self):
        result = knowledge_lookup("NoSuchSchema", "test")
        assert result["total_found"] == 0
        assert "error" in result

    def test_lookup_with_filter(self):
        store = KnowledgeStore(FaultPattern)
        store.insert(FaultPattern(id="F-001", signal_name="engine_temp", root_cause="overheat", solution="Cool down"))
        set_knowledge_store("FaultPattern", store)

        result = knowledge_lookup("FaultPattern", "engine", {"root_cause": "overheat"})
        assert result["total_found"] == 1
        assert result["results"][0]["signal_name"] == "engine_temp"

    def test_lookup_text_search(self):
        store = KnowledgeStore(FaultPattern)
        store.insert(FaultPattern(id="F-002", signal_name="oil_pressure", root_cause="leak", solution="Seal"))
        set_knowledge_store("FaultPattern", store)

        result = knowledge_lookup("FaultPattern", "leak")
        assert result["total_found"] >= 1


# ── Knowledge Schema ───────────────────────────────────────────


def test_knowledge_schema_repr():
    ks = KnowledgeSchema(id="test", created_at=None, updated_at=None)  # type: ignore
    r = repr(ks)
    assert "KnowledgeSchema" in r


# ── Knowledge Store ────────────────────────────────────────────


class TestKnowledgeStore:
    def test_insert_and_get(self):
        store = KnowledgeStore(FaultPattern)
        fp = FaultPattern(id="F-003", signal_name="brake", root_cause="wear", solution="Replace")
        store.insert(fp)
        retrieved = store.get("F-003")
        assert retrieved is not None
        assert retrieved.signal_name == "brake"

    def test_query_no_filters(self):
        store = KnowledgeStore(FaultPattern)
        store.insert(FaultPattern(id="F-004", signal_name="test1", root_cause="a", solution="x"))
        store.insert(FaultPattern(id="F-005", signal_name="test2", root_cause="b", solution="y"))
        results = store.query()
        assert len(results) == 2

    def test_count(self):
        store = KnowledgeStore(FaultPattern)
        assert store.count() == 0
        store.insert(FaultPattern(id="F-006", signal_name="t", root_cause="t", solution="t"))
        assert store.count() == 1


# ── Vector Store ────────────────────────────────────────────────


def test_vector_store_add_search():
    store = VectorStore()
    store.add(["hello world", "goodbye world", "hello python"])
    results = store.search("hello world", top_k=2)
    assert len(results) == 2
    assert results[0]["score"] > 0



def test_vector_store_count_clear():
    store = VectorStore()
    store.add(["doc1", "doc2"])
    assert store.count() == 2
    store.clear()
    assert store.count() == 0


# ── Markdown Adapter ────────────────────────────────────────────


def test_markdown_adapter():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write("# Title\nContent here.\n# Another Title\nMore content.")
        path = f.name

    try:
        chunks = MarkdownAdapter.load(path)
        assert len(chunks) >= 2
        assert any("Title" in c["content"] for c in chunks)
    finally:
        Path(path).unlink()



def test_text_adapter():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("plain text content")
        path = f.name

    try:
        chunks = TextAdapter.load(path)
        assert len(chunks) == 1
        assert "plain text" in chunks[0]["content"]
    finally:
        Path(path).unlink()


# ── Middleware Pipeline ─────────────────────────────────────────


def test_cli_adapter():
    adapter = CLIAdapter()
    assert adapter.state == ConnectionState.CONNECTED
    assert adapter.platform_name == "cli"



def test_telegram_adapter_no_token():
    adapter = TelegramAdapter(bot_token="")
    assert adapter.state == ConnectionState.DISCONNECTED



def test_slack_adapter_no_token():
    adapter = SlackAdapter(bot_token="")
    assert adapter.state == ConnectionState.DISCONNECTED


# ── Phase 3: Training ──────────────────────────────────────────

from kairos.training.recorder import TrajectoryRecorder, ToolContext
from kairos.training.env import (
    TrainingEnv, EnvironmentRegistry, RolloutRunner,
    reward_confidence, reward_success_rate, reward_evidence_quality, reward_file_creation,
)



