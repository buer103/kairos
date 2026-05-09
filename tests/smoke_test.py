"""Integration tests for Kairos framework."""

from __future__ import annotations

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


# ── State ──────────────────────────────────────────────────────

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
from kairos.gateway.adapters.base import CLIAdapter, TelegramAdapter, SlackAdapter


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
    assert "workspace_dir" in state.metadata["thread_data"]
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

class TestCronScheduler:
    def test_register_and_list(self, tmp_path):
        from kairos.cron import CronScheduler, Job, CronSchedule
        db = tmp_path / "cron.db"
        s = CronScheduler(db_path=str(db))
        j = Job(name="daily", schedule=CronSchedule.daily_at(9, 0))
        s.register(j)
        jobs = s.list()
        assert len(jobs) == 1
        assert jobs[0].name == "daily"

    def test_pause_resume(self, tmp_path):
        from kairos.cron import CronScheduler, Job, CronSchedule, JobStatus
        db = tmp_path / "cron.db"
        s = CronScheduler(db_path=str(db))
        j = s.register(Job(name="test"))
        assert s.pause(j.id).status == JobStatus.PAUSED
        assert s.resume(j.id).status == JobStatus.PENDING

    def test_cancel(self, tmp_path):
        from kairos.cron import CronScheduler, Job, CronSchedule, JobStatus
        db = tmp_path / "cron.db"
        s = CronScheduler(db_path=str(db))
        j = s.register(Job(name="test"))
        assert s.cancel(j.id).status == JobStatus.CANCELLED

    def test_remove(self, tmp_path):
        from kairos.cron import CronScheduler, Job
        db = tmp_path / "cron.db"
        s = CronScheduler(db_path=str(db))
        j = s.register(Job(name="removable"))
        s.remove(j.id)
        assert len(s.list()) == 0

    def test_schedule_matches(self):
        from kairos.cron import CronSchedule
        from datetime import datetime, timezone
        sched = CronSchedule.daily_at(9, 0)
        dt = datetime(2026, 1, 5, 9, 0, tzinfo=timezone.utc)
        assert sched.matches(dt)
        assert not sched.matches(datetime(2026, 1, 5, 10, 0, tzinfo=timezone.utc))

    def test_schedule_weekly(self):
        from kairos.cron import CronSchedule
        from datetime import datetime, timezone
        sched = CronSchedule.weekly_on(0, 9, 0)  # Monday 9:00
        mon = datetime(2026, 1, 5, 9, 0, tzinfo=timezone.utc)  # Monday
        tue = datetime(2026, 1, 6, 9, 0, tzinfo=timezone.utc)  # Tuesday
        assert sched.matches(mon)
        assert not sched.matches(tue)

    def test_tick_fires_due_jobs(self, tmp_path):
        from kairos.cron import CronScheduler, Job, CronSchedule
        from datetime import datetime, timezone
        db = tmp_path / "cron.db"
        s = CronScheduler(db_path=str(db))
        now = datetime.now(timezone.utc)
        sched = CronSchedule(minute=[now.minute], hour=[now.hour])
        j = s.register(Job(name="immediate", schedule=sched))
        # Tick fires jobs that match current time
        fired = s.tick()
        assert len(fired) >= 1
        updated = s.get(j.id)
        assert updated.run_count >= 1

    def test_repeat_limit(self, tmp_path):
        from kairos.cron import CronScheduler, Job, CronSchedule
        from datetime import datetime, timezone
        db = tmp_path / "cron.db"
        s = CronScheduler(db_path=str(db))
        now = datetime.now(timezone.utc)
        sched = CronSchedule(minute=[now.minute], hour=[now.hour])
        j = s.register(Job(name="limited", schedule=sched, repeat=1))
        s.tick()
        j = s.get(j.id)
        assert j.run_count == 1
        assert j.status.value == "done"
        # Tick again — should not re-fire (repeat=1 reached)
        fired = s.tick()
        assert len(fired) == 0

    def test_next_fire(self):
        from kairos.cron import CronSchedule
        from datetime import datetime, timezone, timedelta
        sched = CronSchedule.daily_at(12, 0)
        now = datetime(2026, 1, 5, 8, 0, tzinfo=timezone.utc)
        nxt = sched.next_fire(now)
        assert nxt.hour == 12
        assert nxt.day == 5
        assert nxt > now


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


class TestDelegation:
    def test_delegate_task(self):
        from kairos.agents.delegate import DelegateTask
        t = DelegateTask(goal="test", context="test context")
        assert t.goal == "test"
        assert t.context == "test context"
        assert t.id.startswith("subtask_")

    def test_delegate_result(self):
        from kairos.agents.delegate import DelegateResult
        r = DelegateResult(task_id="t1", success=True, content="done")
        assert r.success
        assert r.content == "done"

    def test_delegate_config(self):
        from kairos.agents.delegate import DelegateConfig
        cfg = DelegateConfig(max_concurrent=5, default_timeout=60)
        assert cfg.max_concurrent == 5
        assert cfg.default_timeout == 60

    def test_delegation_manager_init(self):
        from kairos.agents.delegate import DelegationManager, DelegateConfig
        mgr = DelegationManager(model=None, config=DelegateConfig(max_concurrent=2))
        assert mgr.config.max_concurrent == 2

    def test_register_delegate_tool(self):
        from kairos.agents.delegate import DelegationManager, register_delegate_tool, DelegateConfig
        from kairos.tools.registry import get_all_tools
        mgr = DelegationManager(model=None, config=DelegateConfig())
        register_delegate_tool(mgr)
        tools = get_all_tools()  # returns dict[str, dict]
        assert "delegate_task" in tools


# ═══════════════════════════════════════════════════════════════
# Phase 7 — Context Compression v2 + Gateway adapters v2
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


class TestGatewayAdaptersV2:
    def test_discord_init(self):
        from kairos.gateway.adapters import DiscordAdapter
        a = DiscordAdapter(bot_token="test")
        assert a.platform_name == "discord"

    def test_feishu_init(self):
        from kairos.gateway.adapters import FeishuAdapter
        a = FeishuAdapter(app_id="test", app_secret="secret")
        assert a.platform_name == "feishu"

    def test_whatsapp_init(self):
        from kairos.gateway.adapters import WhatsAppAdapter
        a = WhatsAppAdapter(phone_number_id="123", access_token="tok")
        assert a.platform_name == "whatsapp"

    def test_signal_init(self):
        from kairos.gateway.adapters import SignalAdapter
        a = SignalAdapter(sender_number="+1234567890")
        assert a.platform_name == "signal"

    def test_line_init(self):
        from kairos.gateway.adapters import LineAdapter
        a = LineAdapter(channel_access_token="tok")
        assert a.platform_name == "line"

    def test_matrix_init(self):
        from kairos.gateway.adapters import MatrixAdapter
        a = MatrixAdapter(homeserver="https://matrix.org", access_token="tok")
        assert a.platform_name == "matrix"

    def test_irc_init(self):
        from kairos.gateway.adapters import IRCAdapter
        a = IRCAdapter(server="irc.example.com", nickname="kairos-test")
        assert a.platform_name == "irc"

    def test_discord_translate(self):
        from kairos.gateway.adapters import DiscordAdapter
        a = DiscordAdapter()
        # Discord interaction payload format
        msg = a.translate_incoming({
            "id": "msg1",
            "type": 0,  # Not PING
            "channel_id": "C123",
            "guild_id": "G456",
            "author": {"id": "U789", "username": "buer"},
            "data": {"content": "hello kairos"},
        })
        assert msg.platform == "discord"
        assert "hello kairos" in msg.content[0].text
        assert msg.chat_id == "C123"

    def test_whatsapp_translate(self):
        from kairos.gateway.adapters import WhatsAppAdapter
        a = WhatsAppAdapter()
        msg = a.translate_incoming({
            "entry": [{
                "changes": [{
                    "value": {
                        "messages": [{
                            "id": "w1",
                            "from": "551199999999",
                            "type": "text",
                            "text": {"body": "hi from wa"},
                        }],
                        "contacts": [{"profile": {"name": "Test"}}],
                    }
                }]
            }]
        })
        assert msg.platform == "whatsapp"
        assert "hi from wa" in msg.content[0].text
        assert msg.chat_id == "551199999999"

    def test_line_translate(self):
        from kairos.gateway.adapters import LineAdapter
        a = LineAdapter()
        msg = a.translate_incoming({
            "events": [{
                "type": "message",
                "message": {"type": "text", "text": "hello line"},
                "source": {"userId": "U123"},
                "replyToken": "rtok",
                "webhookEventId": "evt1",
            }]
        })
        assert msg.platform == "line"
        assert "hello line" in msg.content[0].text
        assert msg.chat_id == "U123"
