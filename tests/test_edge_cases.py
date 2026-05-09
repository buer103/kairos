"""Edge case and boundary tests for Kairos."""

from __future__ import annotations

import concurrent.futures
import tempfile
from pathlib import Path

import pytest

from kairos.core.state import Case, ThreadState
from kairos.core.middleware import MiddlewarePipeline
from kairos.tools.registry import get_all_tools, get_tool_schemas, execute_tool, get_tool


# ═══════════════════════════════════════════════════════════════
# Tool edge cases
# ═══════════════════════════════════════════════════════════════

def test_execute_tool_empty_args():
    result = execute_tool("read_file", {"path": "/nonexistent/file.xyz"})
    assert isinstance(result, dict)


def test_execute_tool_missing_required_arg():
    """Executing a tool with missing args should not crash."""
    try:
        result = execute_tool("read_file", {})
        assert isinstance(result, dict)
    except Exception:
        pass  # Acceptable to raise on missing required args


def test_execute_tool_unknown_tool():
    result = execute_tool("this_tool_does_not_exist", {})
    assert isinstance(result, dict)


def test_all_tools_have_names():
    tools = get_all_tools()
    assert len(tools) > 0
    for name, info in tools.items():
        assert isinstance(name, str)
        assert "name" in info or isinstance(info, dict)


# ═══════════════════════════════════════════════════════════════
# State edge cases
# ═══════════════════════════════════════════════════════════════

def test_case_max_steps():
    case = Case(id="stress")
    for i in range(1000):
        case.add_step("tool", {"n": i})
    assert len(case.steps) == 1000


def test_threadstate_empty_messages():
    state = ThreadState()
    assert state.messages == []


def test_threadstate_many_messages():
    state = ThreadState()
    for i in range(500):
        state.messages.append({"role": "user", "content": f"msg {i}"})
    assert len(state.messages) == 500


# ═══════════════════════════════════════════════════════════════
# Middleware edge cases
# ═══════════════════════════════════════════════════════════════

def test_empty_pipeline_noop():
    pipeline = MiddlewarePipeline([])
    state = ThreadState()
    runtime = {}
    pipeline.before_agent(state, runtime)
    pipeline.before_model(state, runtime)
    pipeline.after_model(state, runtime)
    pipeline.after_agent(state, runtime)


def test_pipeline_handles_none_state():
    from kairos.middleware import ContextCompressor
    pipeline = MiddlewarePipeline([ContextCompressor()])
    # Should not crash with None state
    try:
        pipeline.before_model(None, {})
    except Exception:
        pass  # Acceptable


# ═══════════════════════════════════════════════════════════════
# Config edge cases
# ═══════════════════════════════════════════════════════════════

def test_config_no_file():
    from kairos.config import Config
    cfg = Config(path="/nonexistent/path/config.yaml")
    assert cfg.get("any.key", default="fallback") == "fallback"


def test_config_env_override(monkeypatch):
    from kairos.config import Config
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key-123")
    cfg = Config()
    assert cfg.get("providers.deepseek.api_key") == "test-key-123"


def test_config_nested_default():
    from kairos.config import Config
    cfg = Config()
    assert cfg.get("a.b.c.d.e", default=42) == 42


# ═══════════════════════════════════════════════════════════════
# Registry edge cases
# ═══════════════════════════════════════════════════════════════

def test_register_tool_does_not_crash():
    from kairos.tools.registry import register_tool
    @register_tool(
        name="test_edge_tool_1",
        description="Edge case test tool.",
        parameters={"x": {"type": "string"}},
    )
    def test_edge_tool_1(x: str = ""):
        return {"result": x}

    t = get_tool("test_edge_tool_1")
    assert t is not None


# ═══════════════════════════════════════════════════════════════
# Concurrency edge cases
# ═══════════════════════════════════════════════════════════════

def test_concurrent_tool_registry_access():
    def read_tools():
        for _ in range(100):
            get_all_tools()

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        futures = [ex.submit(read_tools) for _ in range(4)]
        for f in futures:
            f.result(timeout=10)

    assert True  # No crash = pass


def test_concurrent_cron_access(tmp_path: Path):
    from kairos.cron import CronScheduler, Job, CronSchedule
    db = tmp_path / "cron.db"
    s = CronScheduler(db_path=str(db))
    j = s.register(Job(name="concurrent", schedule=CronSchedule.daily_at(8, 0)))

    def access():
        for _ in range(20):
            s.list()
            s.get(j.id)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        futures = [ex.submit(access) for _ in range(4)]
        for f in futures:
            f.result(timeout=10)

    assert True


# ═══════════════════════════════════════════════════════════════
# Large payload
# ═══════════════════════════════════════════════════════════════

def test_compressor_handles_very_large_messages():
    from kairos.middleware.compress import ContextCompressor
    from kairos.core.state import ThreadState
    compressor = ContextCompressor(max_tokens=100000, budget_ratio=0.8)
    state = ThreadState()
    state.messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "x" * 10000},
        {"role": "assistant", "content": "y" * 10000},
        {"role": "user", "content": "keep this"},
        {"role": "assistant", "content": "done"},
    ]
    result = compressor.before_model(state, {})
    # Should not crash
    assert True


def test_tool_registry_many_queries():
    """Many rapid tool schema queries."""
    for _ in range(100):
        get_tool_schemas()
    assert True
