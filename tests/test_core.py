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


class TestCase:
    def test_create_case(self):
        case = Case(id="test-001")
        assert case.id == "test-001"
        assert case.steps == []

    def test_add_step(self):
        case = Case(id="test-001")
        step = case.add_step("rag_search", {"query": "test"})
        assert step.id == 1
        assert step.tool == "rag_search"
        assert len(case.steps) == 1

    def test_complete_step(self):
        case = Case(id="test-001")
        step = case.add_step("test_tool", {"key": "val"})
        case.complete_step(step, {"result": "ok"}, 42.0)
        assert step.result == {"result": "ok"}
        assert step.duration_ms == 42.0



class TestThreadState:
    def test_default_state(self):
        state = ThreadState()
        assert state.messages == []
        assert state.case is None

    def test_merge_artifacts(self):
        result = merge_artifacts(["a", "b"], ["b", "c"])
        assert result == ["a", "b", "c"]


# ── Tool Registry ───────────────────────────────────────────────


def test_prompt_builder_default():
    pb = PromptBuilder(agent_name="TestBot")
    prompt = pb.build()
    assert "TestBot" in prompt
    assert "You are a helpful AI assistant" in prompt



def test_prompt_builder_custom_soul():
    pb = PromptBuilder(agent_name="Custom", soul="You are a pirate!")
    prompt = pb.build()
    assert "pirate" in prompt


# ── Sub-Agent Types ────────────────────────────────────────────


def test_builtin_types():
    assert "general-purpose" in BUILTIN_TYPES
    assert "bash" in BUILTIN_TYPES
    assert "research" in BUILTIN_TYPES
    assert GENERAL_PURPOSE.max_turns == 50
    assert BASH.max_turns == 30



def test_register_custom_type():
    custom = SubAgentType(name="custom-type", description="Custom", max_turns=10)
    register_subagent_types({"custom-type": custom})
    assert get_subagent_type("custom-type") is not None
    assert get_subagent_type("custom-type").max_turns == 10


# ── Session Store ───────────────────────────────────────────────


def test_unified_message_from_text():
    msg = UnifiedMessage.from_text("Hello world", platform="test")
    assert msg.text == "Hello world"
    assert msg.platform == "test"
    assert msg.role == MessageRole.USER
    assert msg.has_media is False



def test_unified_message_with_media():
    import uuid
    msg = UnifiedMessage(
        id=str(uuid.uuid4())[:12],
        role=MessageRole.USER,
        content=[ContentBlock.text_block("Look at this"), ContentBlock.image_block("/tmp/img.jpg")],
    )
    assert msg.has_media is True
    assert msg.text == "Look at this"



def test_unified_response_to_dict():
    resp = UnifiedResponse(text="Hello", confidence=0.95, evidence=[{"step": 1}])
    d = resp.to_dict()
    assert d["text"] == "Hello"
    assert d["confidence"] == 0.95
    assert len(d["evidence"]) == 1



def test_content_block_types():
    assert ContentBlock.text_block("text").type == ContentType.TEXT
    assert ContentBlock.image_block("url").type == ContentType.IMAGE



def test_trajectory_recorder(tmp_path):
    rec = TrajectoryRecorder(output_dir=tmp_path)
    rec.record(
        [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi!"}],
        metadata={"confidence": 0.9},
    )
    assert rec.total_recorded() == 1
    path = rec.flush()
    assert path.exists()
    content = path.read_text()
    assert "Hello" in content
    assert "confidence" in content



def test_trajectory_recorder_buffering(tmp_path):
    rec = TrajectoryRecorder(output_dir=tmp_path)
    rec.record([{"role": "user", "content": "A"}])
    rec.record([{"role": "user", "content": "B"}])
    assert rec.total_recorded() == 2
    assert rec.count() == 2
    rec.flush()
    assert rec.count() == 0
    assert rec.total_recorded() == 2



def test_tool_context_file_created(tmp_path):
    ctx = ToolContext(workdir=tmp_path)
    ctx.snapshot_before()
    (tmp_path / "output.txt").write_text("result")
    ctx.snapshot_after()
    assert ctx.file_created("output.txt")
    assert ctx.file_changed("output.txt")
    assert not ctx.file_created("nonexistent.txt")



def test_tool_context_grep():
    ctx = ToolContext()
    ctx.record_terminal("error: segmentation fault at 0x1234")
    ctx.record_terminal("file written to /tmp/output")
    assert ctx.grep_output("segmentation fault")
    assert ctx.grep_output("written")
    assert not ctx.grep_output("nonexistent")



def test_training_env():
    env = TrainingEnv(
        name="test-env",
        prompt_template="Analyze {source}",
        reward_fn=lambda r, c: 0.75,
    )
    assert env.format_prompt(source="log.txt") == "Analyze log.txt"
    assert env.evaluate({}, ToolContext()) == 0.75



def test_environment_registry():
    reg = EnvironmentRegistry()
    env = TrainingEnv(name="env-a")
    reg.register(env)
    assert reg.get("env-a") is not None
    assert reg.get("nonexistent") is None
    assert "env-a" in reg.list_envs()



def test_retry_config():
    cfg = RetryConfig(max_retries=3, base_delay=1.0)
    delay = cfg.delay_for_attempt(0)
    assert delay > 0
    delay2 = cfg.delay_for_attempt(1)
    assert delay2 > delay  # Backoff increases


# ── P0/P1: LLM Retry Middleware ─────────────────────────────────

from kairos.middleware.llm_retry import LLMRetryMiddleware, ToolArgRepairMiddleware



def test_tool_arg_repair_trailing_comma():
    mw = ToolArgRepairMiddleware()
    repaired = mw._repair_json('{"key": "val",}')
    assert repaired == {"key": "val"}



def test_tool_arg_repair_single_quotes():
    mw = ToolArgRepairMiddleware()
    repaired = mw._repair_json("{'key': 'val'}")
    assert repaired == {"key": "val"}



def test_tool_arg_repair_python_bool():
    mw = ToolArgRepairMiddleware()
    repaired = mw._repair_json('{"key": True, "other": False, "none": None}')
    assert repaired == {"key": True, "other": False, "none": None}



def test_tool_arg_repair_valid_passthrough():
    mw = ToolArgRepairMiddleware()
    result = mw.wrap_tool_call("test", {"key": "val"}, lambda n, a, **kw: a)
    assert result == {"key": "val"}


# ── P0/P1: Stateful Agent ──────────────────────────────────────

from kairos.core.stateful_agent import StatefulAgent
from kairos.providers.base import ModelConfig



