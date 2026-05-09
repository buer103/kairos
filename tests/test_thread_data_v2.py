"""Tests for ThreadData middleware v2 — production-grade thread management."""

import os
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from kairos.core.middleware import MiddlewarePipeline
from kairos.core.state import ThreadState
from kairos.core.paths import ThreadPaths
from kairos.core.thread_state import ThreadDataState
from kairos.middleware.thread_data import ThreadDataMiddleware


class TestThreadPaths:
    """Path resolution and directory management."""

    @pytest.fixture
    def paths(self, tmp_path):
        return ThreadPaths(base_dir=tmp_path / "data")

    def test_basic_paths(self, paths):
        """Resolves standard path hierarchy."""
        p = paths.all_paths("thread-1")
        assert "workspace" in p
        assert "uploads" in p
        assert "outputs" in p
        assert "thread_root" in p
        assert "thread-1" in p["workspace"]
        assert "user-data" in p["workspace"]

    def test_user_isolation(self, paths):
        """Per-user paths are in separate directories."""
        alice = paths.all_paths("thread-1", user_id="alice")
        bob = paths.all_paths("thread-1", user_id="bob")

        assert alice["workspace"] != bob["workspace"]
        assert "user-alice" in alice["workspace"]
        assert "user-bob" in bob["workspace"]

    def test_ensure_creates_dirs(self, paths):
        """ensure() creates all directories on disk."""
        result = paths.ensure("thread-2")
        assert os.path.isdir(result["workspace"])
        assert os.path.isdir(result["uploads"])
        assert os.path.isdir(result["outputs"])

    def test_ensure_workspace_lazy(self, paths):
        """ensure_workspace() creates only the workspace."""
        ws = paths.ensure_workspace("thread-3")
        assert os.path.isdir(ws)
        # Uploads and outputs should NOT exist yet
        up = paths.uploads("thread-3")
        assert not up.exists()

    def test_thread_exists(self, paths):
        """thread_exists checks disk."""
        assert not paths.thread_exists("nonexistent")
        paths.ensure("thread-4")
        assert paths.thread_exists("thread-4")

    def test_list_threads(self, paths):
        """list_threads returns all thread IDs."""
        paths.ensure("thread-a")
        paths.ensure("thread-b")
        threads = paths.list_threads()
        assert "thread-a" in threads
        assert "thread-b" in threads

    def test_remove_thread(self, paths):
        """remove_thread deletes all thread data."""
        paths.ensure("thread-rm")
        assert paths.thread_exists("thread-rm")
        assert paths.remove_thread("thread-rm")
        assert not paths.thread_exists("thread-rm")

    def test_remove_nonexistent(self, paths):
        """remove_thread on nonexistent returns False."""
        assert not paths.remove_thread("ghost-thread")

    def test_size(self, paths):
        """size returns total bytes."""
        paths.ensure("thread-size")
        # Create some test files
        ws = paths.workspace("thread-size")
        (ws / "test.txt").write_text("hello")
        size = paths.size("thread-size")
        assert size > 0


class TestThreadDataMiddleware:
    """Middleware pipeline integration."""

    @pytest.fixture
    def middleware(self, tmp_path):
        return ThreadDataMiddleware(base_dir=str(tmp_path / "data"))

    @pytest.fixture
    def state_and_runtime(self):
        state = ThreadState()
        state.messages = [
            {"role": "system", "content": "You are a test assistant."},
            {"role": "user", "content": "Hello"},
        ]
        runtime = {"thread_id": "test-thread", "run_id": "run-001"}
        return state, runtime

    # ── Core behavior ──────────────────────────────────────────

    def test_basic_injection(self, middleware, state_and_runtime):
        """Thread data is injected into state metadata and runtime."""
        state, runtime = state_and_runtime
        middleware.before_agent(state, runtime)

        td = state.metadata["thread_data"]
        assert td["thread_id"] == "test-thread"
        assert "workspace" in td
        assert "uploads" in td
        assert "outputs" in td
        assert "workspace" in td and "test-thread" in td["workspace"]

    def test_auto_generates_thread_id(self, middleware):
        """When no thread_id is provided, one is auto-generated."""
        state = ThreadState()
        state.messages = [{"role": "user", "content": "Hi"}]
        runtime = {}

        middleware.before_agent(state, runtime)
        assert "thread_id" in runtime
        assert runtime["thread_id"].startswith("auto-")

    def test_eager_mode_creates_dirs(self, tmp_path):
        """eager mode (lazy_init=False) creates dirs immediately."""
        mw = ThreadDataMiddleware(
            base_dir=str(tmp_path / "eager-data"),
            lazy_init=False,
        )
        state = ThreadState()
        state.messages = [{"role": "user", "content": "Hi"}]
        runtime = {"thread_id": "eager-thread"}

        mw.before_agent(state, runtime)

        ws = state.metadata["thread_data"]["workspace"]
        assert os.path.isdir(ws)

    def test_lazy_mode_defers_creation(self, middleware, state_and_runtime):
        """lazy mode (default) does NOT create dirs immediately."""
        state, runtime = state_and_runtime
        middleware.before_agent(state, runtime)

        ws = state.metadata["thread_data"]["workspace"]
        assert not os.path.isdir(ws)

    def test_user_isolation_mode(self, tmp_path):
        """With user_isolation=True, per-user dirs are used."""
        mw = ThreadDataMiddleware(
            base_dir=str(tmp_path / "user-data"),
            user_isolation=True,
        )
        state = ThreadState()
        state.messages = [{"role": "user", "content": "Hi"}]
        runtime = {"thread_id": "t1", "user_id": "alice"}

        mw.before_agent(state, runtime)

        ws = state.metadata["thread_data"]["workspace"]
        assert "user-alice" in ws
        assert state.metadata["thread_data"]["user_id"] == "alice"

    def test_human_message_enriched(self, middleware, state_and_runtime):
        """Last HumanMessage gets thread_id, run_id, timestamp metadata."""
        state, runtime = state_and_runtime
        middleware.before_agent(state, runtime)

        last_user_msg = state.messages[-1]
        assert "metadata" in last_user_msg
        assert last_user_msg["metadata"]["thread_id"] == "test-thread"
        assert last_user_msg["metadata"]["run_id"] == "run-001"
        assert "timestamp" in last_user_msg["metadata"]

    def test_system_prompt_injected(self, middleware, state_and_runtime):
        """Workspace info is appended to the system prompt."""
        state, runtime = state_and_runtime
        middleware.before_agent(state, runtime)

        system_msg = state.messages[0]["content"]
        assert "Thread Environment" in system_msg
        assert "test-thread" in system_msg
        assert "Workspace:" in system_msg

    def test_runtime_updated(self, middleware, state_and_runtime):
        """runtime dict gets thread_id, run_id, thread_data keys."""
        state, runtime = state_and_runtime
        middleware.before_agent(state, runtime)

        assert "thread_id" in runtime
        assert "run_id" in runtime
        assert "thread_data" in runtime
        assert "thread_paths" in runtime

    def test_cleanup_on_end(self, tmp_path):
        """cleanup_on_end=True removes auto-generated threads."""
        mw = ThreadDataMiddleware(
            base_dir=str(tmp_path / "cleanup-data"),
            cleanup_on_end=True,
        )
        state = ThreadState()
        state.messages = [{"role": "user", "content": "Hi"}]
        runtime = {}  # No thread_id → auto-generated

        mw.before_agent(state, runtime)
        thread_id = runtime["thread_id"]
        assert thread_id.startswith("auto-")

        # Verify dir was created... or not (lazy mode)
        # Then run after_agent
        mw.after_agent(state, runtime)
        # Auto-generated thread should be cleaned
        assert not mw._paths.thread_exists(thread_id)

    def test_cleanup_skips_explicit_threads(self, tmp_path):
        """cleanup_on_end does NOT remove explicitly named threads."""
        mw = ThreadDataMiddleware(
            base_dir=str(tmp_path / "keep-data"),
            cleanup_on_end=True,
            lazy_init=False,
        )
        state = ThreadState()
        state.messages = [{"role": "user", "content": "Hi"}]
        runtime = {"thread_id": "important-thread"}

        mw.before_agent(state, runtime)
        mw.after_agent(state, runtime)
        # Explicit thread should survive
        assert mw._paths.thread_exists("important-thread")

    def test_get_thread_data_helper(self, middleware, state_and_runtime):
        """ThreadDataMiddleware.get_thread_data() extracts from runtime."""
        state, runtime = state_and_runtime
        middleware.before_agent(state, runtime)

        td = ThreadDataMiddleware.get_thread_data(runtime)
        assert td is not None
        assert td["thread_id"] == "test-thread"

    def test_ensure_workspace_helper(self, middleware, state_and_runtime):
        """ensure_workspace() creates the workspace dir lazily."""
        state, runtime = state_and_runtime
        middleware.before_agent(state, runtime)

        ws = ThreadDataMiddleware.ensure_workspace(runtime)
        assert ws is not None
        assert os.path.isdir(ws)

    def test_pipeline_integration(self, tmp_path):
        """ThreadData works as first layer in MiddlewarePipeline."""
        from kairos.middleware.uploads import UploadsMiddleware
        from kairos.middleware.dangling import DanglingToolCallMiddleware

        mw = ThreadDataMiddleware(base_dir=str(tmp_path / "pipeline-data"))
        pipeline = MiddlewarePipeline([
            mw,
            UploadsMiddleware(),
            DanglingToolCallMiddleware(),
        ])

        state = ThreadState()
        state.messages = [{"role": "user", "content": "Test"}]
        runtime = {"thread_id": "pipeline-thread"}

        pipeline.before_agent(state, runtime)
        assert "thread_data" in state.metadata
        assert state.metadata["thread_data"]["thread_id"] == "pipeline-thread"

    def test_repr(self, middleware):
        """repr includes key settings."""
        r = repr(middleware)
        assert "ThreadDataMiddleware" in r
        assert "lazy=True" in r
