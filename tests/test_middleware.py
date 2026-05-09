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


def test_empty_pipeline():
    pipeline = MiddlewarePipeline([])
    state = ThreadState()
    runtime = {}
    # Should not raise
    pipeline.before_agent(state, runtime)
    pipeline.after_agent(state, runtime)



def test_evidence_tracker():
    tracker = EvidenceTracker()
    case = Case(id="test")
    state = ThreadState(case=case)

    result = tracker.wrap_tool_call(
        "test_tool",
        {"key": "val"},
        lambda name, args, **kw: {"ok": True},
        state=state,
    )
    assert result == {"ok": True}
    assert len(case.steps) == 1
    assert case.steps[0].tool == "test_tool"



class TestConfidenceScorer:
    def test_no_steps(self):
        scorer = ConfidenceScorer()
        case = Case(id="test")
        state = ThreadState(case=case)
        scorer.after_agent(state, {})
        assert case.confidence is None  # No steps → no score

    def test_with_steps(self):
        scorer = ConfidenceScorer()
        case = Case(id="test")
        state = ThreadState(case=case)
        step = case.add_step("t1", {"a": 1})
        case.complete_step(step, {"result": "good"}, 10.0)
        scorer.after_agent(state, {})
        assert case.confidence is not None
        assert 0 < case.confidence <= 0.99



def test_context_compressor():
    compressor = ContextCompressor(max_tokens=200, budget_ratio=0.5, keep_recent=2)
    state = ThreadState()
    state.messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "a" * 200},
        {"role": "assistant", "content": "b" * 200},
        {"role": "user", "content": "c" * 200},
        {"role": "assistant", "content": "d" * 200},
        {"role": "user", "content": "e" * 200},
        {"role": "assistant", "content": "End."},
    ]
    runtime = {}
    result = compressor.before_model(state, runtime)
    # With budget_ratio=0.5, max=200, budget=~100-1024=negative, so it should compress
    assert result is not None
    assert result["compressed_after"] < result["compressed_before"]


# ── Skill Loader ────────────────────────────────────────────────


def test_skill_loader_no_dir():
    loader = SkillLoader(skills_dir="/tmp/nonexistent-kairos-skills")
    state = ThreadState()
    runtime = {}
    loader.before_agent(state, runtime)
    assert len(loader.get_skills()) == 0



def test_skill_loader_with_skills(tmp_path):
    skills_dir = tmp_path / "skills" / "my-skill"
    skills_dir.mkdir(parents=True)
    (skills_dir / "SKILL.md").write_text(
        "---\n"
        "name: test-skill\n"
        "description: A test skill\n"
        "---\n"
        "# Test Skill\n"
        "This is a test.\n"
    )

    loader = SkillLoader(skills_dir=skills_dir.parent)
    state = ThreadState()
    runtime = {}
    loader.before_agent(state, runtime)
    skills = loader.get_skills()
    assert len(skills) == 1
    assert skills[0].name == "test-skill"
    assert skills[0].description == "A test skill"

    # Test find
    found = loader.find_skill("test-skill")
    assert found is not None
    assert found.name == "test-skill"


# ── System Prompt ───────────────────────────────────────────────


def test_evidence_db_save_load(tmp_path):
    db = EvidenceDB(base_path=tmp_path)
    case = Case(id="case-1")
    case.add_step("rag_search", {"query": "test"})
    case.conclusion = "All clear"
    case.confidence = 0.95

    db.save(case)

    loaded = db.load("case-1")
    assert loaded is not None
    assert loaded.conclusion == "All clear"
    assert loaded.confidence == 0.95
    assert len(loaded.steps) == 1

    cases = db.list_cases()
    assert len(cases) == 1


# ── Phase 2: Memory Store ──────────────────────────────────────

from kairos.memory.store import MemoryStore
from kairos.memory.middleware import MemoryMiddleware



def test_reward_confidence():
    assert reward_confidence({"confidence": 0.5}, ToolContext()) == 0.5
    assert reward_confidence({"confidence": None}, ToolContext()) == 0.0
    assert reward_confidence({}, ToolContext()) == 0.0



def test_reward_success_rate():
    assert reward_success_rate({"content": "long enough"}, ToolContext()) == 1.0
    assert reward_success_rate({"content": "ab"}, ToolContext()) == 0.0
    assert reward_success_rate({}, ToolContext()) == 0.0



def test_reward_evidence_quality():
    assert reward_evidence_quality({"evidence": [1, 2, 3, 4, 5]}, ToolContext()) == 1.0
    assert reward_evidence_quality({"evidence": [1]}, ToolContext()) == 0.2
    assert reward_evidence_quality({}, ToolContext()) == 0.0



def test_reward_file_creation(tmp_path):
    ctx = ToolContext(workdir=tmp_path)
    ctx.snapshot_before()
    (tmp_path / "output.txt").write_text("data")
    ctx.snapshot_after()
    assert reward_file_creation({}, ctx) == 1.0

    ctx2 = ToolContext(workdir=tmp_path)
    ctx2.snapshot_before()
    ctx2.snapshot_after()
    assert reward_file_creation({}, ctx2) == 0.0


# ── Middleware Expansion (DeerFlow layers) ──────────────────────

from kairos.middleware.dangling import DanglingToolCallMiddleware
from kairos.middleware.subagent_limit import SubagentLimitMiddleware
from kairos.middleware.clarify import ClarificationMiddleware
from kairos.middleware.thread_data import ThreadDataMiddleware
from kairos.middleware.todo import TodoMiddleware
from kairos.middleware.title import TitleMiddleware
from kairos.middleware.uploads import UploadsMiddleware
from kairos.middleware.view_image import ViewImageMiddleware



def test_dangling_tool_call_fix():
    mw = DanglingToolCallMiddleware()
    messages = [
        {"role": "user", "content": "test"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "tc1", "function": {"name": "view_image", "arguments": "{}"}},
        ]},
    ]
    patched = mw._patch(messages)
    # Should have inserted a synthetic ToolMessage
    assert any(m.get("tool_call_id") == "tc1" for m in patched)
    assert any("interrupted" in m.get("content", "").lower() for m in patched)



def test_dangling_tool_call_no_dangle():
    mw = DanglingToolCallMiddleware()
    messages = [
        {"role": "user", "content": "test"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "tc1", "function": {"name": "search", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "tc1", "content": "result"},
    ]
    patched = mw._patch(messages)
    assert patched == messages  # No changes



def test_clarification_intercept():
    mw = ClarificationMiddleware()
    result = mw.wrap_tool_call(
        "ask_user",
        {"question": "Which file should I analyze?", "options": ["a.txt", "b.txt"]},
        lambda n, a, **kw: {"ok": True},
    )
    assert result.get("clarification") is True
    assert "a.txt" in result.get("formatted", "")
    assert mw.is_clarifying
    assert "Which file" in mw.question



def test_clarification_pass_through():
    mw = ClarificationMiddleware()
    result = mw.wrap_tool_call(
        "rag_search", {"query": "test"},
        lambda n, a, **kw: {"results": []},
    )
    assert result == {"results": []}  # Non-clarification tools pass through



def test_thread_data_middleware():
    import tempfile, os
    td = tempfile.mkdtemp()
    mw = ThreadDataMiddleware(base_dir=td, lazy_init=True)
    from kairos.core.state import ThreadState
    state = ThreadState()
    state.messages = [{"role": "system", "content": "You are helpful."}]
    runtime = {"thread_id": "test-thread"}
    mw.before_agent(state, runtime)
    assert "thread_data" in state.metadata
    assert "workspace" in state.metadata["thread_data"]
    assert "Workspace" in state.messages[0]["content"]



def test_todo_middleware_set_get():
    mw = TodoMiddleware()
    todos = [
        {"content": "Analyze logs", "status": "pending"},
        {"content": "Write report", "status": "pending"},
    ]
    mw.set_todos(todos)
    assert len(mw.get_todos()) == 2
    assert mw.get_todos()[0]["content"] == "Analyze logs"



def test_title_middleware(tmp_path):
    import tempfile, os
    mw = TitleMiddleware()
    from kairos.core.state import ThreadState
    state = ThreadState()
    state.messages = [
        {"role": "user", "content": "Help me fix the segmentation fault in my C++ code"},
    ]
    runtime = {}
    mw.after_agent(state, runtime)
    assert mw.title == "Help me fix the segmentation fault in my C++ code"
    assert runtime.get("title") == mw.title



def test_title_middleware_truncation():
    mw = TitleMiddleware()
    from kairos.core.state import ThreadState
    state = ThreadState()
    state.messages = [
        {"role": "user", "content": "A" * 100 + "\nmore text here"},
    ]
    mw.after_agent(state, {})
    assert len(mw.title) <= TitleMiddleware.MAX_TITLE_LENGTH



def test_uploads_middleware_no_dir():
    mw = UploadsMiddleware()
    from kairos.core.state import ThreadState
    state = ThreadState()
    state.messages = [{"role": "user", "content": "test"}]
    mw.before_agent(state, {})
    assert state.messages[0]["content"] == "test"  # Unchanged



def test_view_image_no_vision():
    mw = ViewImageMiddleware(supports_vision=False)
    from kairos.core.state import ThreadState
    state = ThreadState()
    state.messages = [{"role": "user", "content": "test"}]
    mw.before_model(state, {})
    assert len(state.messages) == 1  # No injection when vision disabled


# ── P0/P1: Credential Pool ─────────────────────────────────────

from kairos.providers.credential import CredentialPool, Credential, RetryConfig



def test_credential_pool_add_acquire():
    pool = CredentialPool()
    pool.add("sk-abc", provider="openai", label="key1")
    pool.add("sk-def", provider="openai", label="key2")
    cred = pool.acquire("openai")
    assert cred is not None
    assert cred.key == "sk-abc"  # First added, fewest failures



def test_credential_pool_release():
    pool = CredentialPool()
    cred = pool.add("sk-test", provider="openai")
    pool.release(cred, success=True)
    assert cred.consecutive_failures == 0
    pool.release(cred, success=False)
    assert cred.consecutive_failures == 1



def test_credential_pool_rate_limit():
    pool = CredentialPool()
    cred = pool.add("sk-test", provider="openai")
    pool.mark_rate_limited(cred, retry_after=0.1)
    assert not cred.available
    import time
    time.sleep(0.15)
    assert cred.available



def test_credential_pool_stats():
    pool = CredentialPool()
    pool.add("sk-1", provider="openai")
    pool.add("sk-2", provider="openai")
    stats = pool.stats("openai")
    assert stats["openai"]["total_keys"] == 2
    assert stats["openai"]["available_keys"] == 2



def test_stateful_agent_save_load(tmp_path):
    agent = StatefulAgent(model=ModelConfig(api_key="test-key"))
    agent._session_dir = tmp_path

    # Mock state for testing
    from kairos.core.state import ThreadState, Case
    agent._state = ThreadState()
    agent._state.messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
    ]

    path = agent.save_session("test-session")
    assert path.exists()

    sessions = agent.list_sessions()
    assert len(sessions) == 1
    assert sessions[0]["name"] == "test-session"



def test_stateful_agent_load_session(tmp_path):
    agent = StatefulAgent(model=ModelConfig(api_key="test-key"))
    agent._session_dir = tmp_path

    # Save first
    from kairos.core.state import ThreadState
    agent._state = ThreadState()
    agent._state.messages = [{"role": "user", "content": "Hi"}]
    agent.save_session("load-test")

    # Load into new agent
    agent2 = StatefulAgent(model=ModelConfig(api_key="test-key"))
    agent2._session_dir = tmp_path
    assert agent2.load_session("load-test")
    assert agent2.history == [{"role": "user", "content": "Hi"}]



def test_stateful_agent_delete_session(tmp_path):
    agent = StatefulAgent(model=ModelConfig(api_key="test-key"))
    agent._session_dir = tmp_path

    from kairos.core.state import ThreadState
    agent._state = ThreadState()
    agent._state.messages = [{"role": "user", "content": "x"}]
    agent.save_session("del-test")

    assert agent.delete_session("del-test") is True
    assert agent.delete_session("nonexistent") is False
    assert len(agent.list_sessions()) == 0



def test_stateful_agent_reset():
    agent = StatefulAgent(model=ModelConfig(api_key="test-key"))
    sid1 = agent.session_id
    agent.reset()
    assert agent.session_id != sid1
    assert agent.turn_count == 0



def test_stateful_agent_interrupt(tmp_path):
    agent = StatefulAgent(model=ModelConfig(api_key="test-key"))
    agent._session_dir = tmp_path
    from kairos.core.state import ThreadState
    agent._state = ThreadState()
    agent._state.messages = [{"role": "user", "content": "hi"}]
    agent.interrupt()
    result = agent.chat("hello")
    assert result["content"] == "[Interrupted]"


# ═══════════════════════════════════════════════════════════════
# Phase 6 — Cron, Rich TUI, Sandbox wiring, Delegation
# ═══════════════════════════════════════════════════════════════


class TestContextCompressorV2:
    def test_passthrough_when_under_budget(self):
        from kairos.middleware.compress import ContextCompressor
        from kairos.core.state import ThreadState
        compressor = ContextCompressor(max_tokens=100000, budget_ratio=0.85)
        state = ThreadState()
        state.messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        result = compressor.before_model(state, {})
        assert result is None  # Passthrough

    def test_tool_output_truncation(self):
        from kairos.middleware.compress import ContextCompressor
        from kairos.core.state import ThreadState
        compressor = ContextCompressor(
            max_tokens=1000, budget_ratio=0.5,
            tool_truncate=50, keep_recent=2,
        )
        state = ThreadState()
        state.messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "test " * 50},
            {"role": "assistant", "content": "ok", "tool_calls": [
                {"id": "1", "function": {"name": "read", "arguments": "{}"}}
            ]},
            {"role": "tool", "content": "x" * 500},  # Long tool output
            {"role": "assistant", "content": "done"},
        ]
        result = compressor.before_model(state, {})
        assert result is not None
        assert result["compressed_after"] < result["compressed_before"]

    def test_layered_summary(self):
        from kairos.middleware.compress import ContextCompressor
        from kairos.core.state import ThreadState
        compressor = ContextCompressor(
            max_tokens=2000, budget_ratio=0.5, keep_recent=2,
            importance_scoring=True,
        )
        state = ThreadState()
        # Large messages to trigger compression
        big = "x" * 300
        state.messages = [
            {"role": "system", "content": "helpful"},
            {"role": "user", "content": big},
            {"role": "assistant", "content": big},
            {"role": "user", "content": big},
            {"role": "assistant", "content": big},
            {"role": "user", "content": big},
            {"role": "assistant", "content": big},
            {"role": "user", "content": "keep this question?"},
            {"role": "assistant", "content": "final answer"},
        ]
        result = compressor.before_model(state, {})
        assert result is not None
        # The last 2 messages should still be there
        assert any("keep this question" in str(m.get("content", "")) for m in state.messages)

    def test_tool_output_keeps_errors(self):
        from kairos.middleware.compress import ContextCompressor
        from kairos.core.state import ThreadState
        compressor = ContextCompressor(
            max_tokens=1000, budget_ratio=0.5, tool_truncate=100,
        )
        state = ThreadState()
        content = (
            "line1\nline2\nline3\n" + "data " * 50
            + "\nERROR: permission denied\n"
            + "last line 1\nlast line 2\nlast line 3"
        )
        state.messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "test " * 50},
            {"role": "tool", "content": content},
            {"role": "user", "content": "test " * 50},
            {"role": "assistant", "content": "done"},
        ]
        result = compressor.before_model(state, {})
        # The ERROR line should be preserved
        tool_msgs = [m for m in state.messages if m.get("role") == "tool"]
        if tool_msgs:
            assert "ERROR" in tool_msgs[0]["content"]

    def test_token_count(self):
        from kairos.middleware.compress import count_tokens
        assert count_tokens("hello world") > 0
        assert count_tokens("你好世界") > 0
        assert count_tokens("") == 0

    def test_stats_tracking(self):
        from kairos.middleware.compress import ContextCompressor
        from kairos.core.state import ThreadState
        compressor = ContextCompressor(
            max_tokens=200, budget_ratio=0.5, keep_recent=2,
            track_compression=True,
        )
        state = ThreadState()
        state.messages = [
            {"role": "user", "content": "x" * 100},
            {"role": "assistant", "content": "y" * 100},
            {"role": "user", "content": "z" * 100},
            {"role": "assistant", "content": "done"},
        ]
        compressor.before_model(state, {})
        assert len(compressor.stats) >= 1
        assert "ratio_pct" in compressor.stats[0]



