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


def test_skill_manager_create(tmp_path):
    sm = SkillManager(skills_dir=tmp_path)
    path = sm.create("test-skill", "# Hello\nWorld", description="A test skill")
    assert path.exists()
    assert path.name == "SKILL.md"
    content = path.read_text()
    assert "name: test-skill" in content
    assert "description: A test skill" in content
    assert "# Hello" in content



def test_skill_manager_update(tmp_path):
    sm = SkillManager(skills_dir=tmp_path)
    sm.create("test-skill", "# Old")
    sm.update("test-skill", content="# New\nBetter content", description="Updated desc")
    entry = sm.get("test-skill")
    assert entry is not None
    content = entry.path.read_text()
    assert "# New" in content



def test_skill_manager_lifecycle(tmp_path):
    sm = SkillManager(skills_dir=tmp_path)
    sm.create("lifecycle-skill", "# Lifecycle Test")
    sm.mark_used("lifecycle-skill")
    entry = sm.get("lifecycle-skill")
    assert entry.use_count == 1
    assert entry.status == SkillStatus.ACTIVE



def test_skill_manager_delete_and_forward(tmp_path):
    sm = SkillManager(skills_dir=tmp_path)
    sm.create("old-skill", "# Old")
    sm.delete("old-skill", absorbed_into="new-skill")
    stats = sm.stats()
    assert stats["archived"] == 1
    target = sm.resolve_forwarding("old-skill")
    assert target == "new-skill"



def test_skill_manager_stats(tmp_path):
    sm = SkillManager(skills_dir=tmp_path)
    sm.create("s1", "# Test")
    sm.create("s2", "# Test 2")
    stats = sm.stats()
    assert stats["active"] == 2
    assert stats["total"] == 2



def test_skill_manager_categories(tmp_path):
    sm = SkillManager(skills_dir=tmp_path)
    sm.create("skill-a", "# A", category="cat1")
    sm.create("skill-b", "# B", category="cat2")
    cats = sm.list_categories()
    assert "cat1" in cats
    assert "cat2" in cats



def test_skill_manager_load_content(tmp_path):
    sm = SkillManager(skills_dir=tmp_path)
    sm.create("load-test", "# Loaded\nThis is content.", description="A load test")
    skill = sm.load_skill_content("load-test")
    assert skill is not None
    assert skill.name == "load-test"
    assert "Loaded" in skill.content


# ── Phase 2: Session Search ────────────────────────────────────

from kairos.session.search import SessionSearch



