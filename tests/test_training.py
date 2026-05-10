"""Tests for Kairos training layer: TrajectoryRecorder, ToolContext, TrainingEnv,
EnvironmentRegistry, RolloutRunner, and built-in reward functions.

Covers: 3 files (recorder.py, env.py) — 374 lines total, currently 0 tests.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from kairos.training.recorder import TrajectoryRecorder, ToolContext
from kairos.training.env import (
    TrainingEnv, EnvironmentRegistry, RolloutRunner,
    reward_confidence, reward_success_rate, reward_evidence_quality,
    reward_file_creation,
)


# ============================================================================
# TrajectoryRecorder
# ============================================================================

class TestTrajectoryRecorder:
    """Tests for the ShareGPT JSONL trajectory recorder."""

    @pytest.fixture
    def recorder(self):
        with tempfile.TemporaryDirectory() as tmp:
            yield TrajectoryRecorder(output_dir=tmp)

    def test_init_creates_output_dir(self, recorder):
        assert recorder._output_dir.exists()
        assert recorder._output_dir.is_dir()

    def test_record_simple_conversation(self, recorder):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        recorder.record(messages)
        assert recorder.count() == 1
        assert recorder.total_recorded() == 1

    def test_record_assigns_sequential_ids(self, recorder):
        recorder.record([{"role": "user", "content": "A"}])
        recorder.record([{"role": "user", "content": "B"}])
        # Check entry IDs
        assert recorder._buffer[0]["id"] == "traj_000000"
        assert recorder._buffer[1]["id"] == "traj_000001"

    def test_record_converts_roles(self, recorder):
        messages = [
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Answer"},
        ]
        recorder.record(messages)

        convs = recorder._buffer[0]["conversations"]
        assert convs[0] == {"from": "human", "value": "Question"}
        assert convs[1] == {"from": "gpt", "value": "Answer"}

    def test_record_converts_tool_role(self, recorder):
        messages = [
            {"role": "user", "content": "Read file"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "function": {"name": "read_file", "arguments": '{"path":"/f"}'}}
            ]},
            {"role": "tool", "content": "file contents"},
            {"role": "assistant", "content": "Done"},
        ]
        recorder.record(messages)

        convs = recorder._buffer[0]["conversations"]
        assert convs[2] == {"from": "tool", "value": "file contents"}

    def test_record_skips_system(self, recorder):
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hi"},
        ]
        recorder.record(messages)
        convs = recorder._buffer[0]["conversations"]
        # system should be skipped, only human remains
        assert len(convs) == 1
        assert convs[0]["from"] == "human"

    def test_record_includes_tool_calls_in_gpt_message(self, recorder):
        messages = [
            {"role": "user", "content": "Do X"},
            {"role": "assistant", "content": "Let me check", "tool_calls": [
                {"function": {"name": "search", "arguments": '{"q":"test"}'}}
            ]},
        ]
        recorder.record(messages)
        convs = recorder._buffer[0]["conversations"]
        gpt_msg = convs[1]
        assert gpt_msg["from"] == "gpt"
        assert "<tool_calls>" in gpt_msg["value"]
        assert "search" in gpt_msg["value"]

    def test_record_includes_metadata(self, recorder):
        messages = [{"role": "user", "content": "Test"}]
        metadata = {"confidence": 0.95, "duration_ms": 1234}
        recorder.record(messages, metadata=metadata)
        assert recorder._buffer[0]["metadata"] == metadata

    def test_record_includes_tools(self, recorder):
        messages = [{"role": "user", "content": "Test"}]
        tools = [{"type": "function", "function": {"name": "test_tool"}}]
        recorder.record(messages, tools=tools)
        assert recorder._buffer[0]["tools"] == tools

    def test_flush_writes_jsonl(self, recorder):
        recorder.record(
            [{"role": "user", "content": "Hello"}],
            metadata={"confidence": 0.9},
        )
        path = recorder.flush("test.jsonl")
        assert path.exists()
        content = path.read_text().strip()
        lines = content.split("\n")
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["id"] == "traj_000000"

    def test_flush_clears_buffer(self, recorder):
        recorder.record([{"role": "user", "content": "A"}])
        recorder.flush()
        assert recorder.count() == 0

    def test_flush_empty_buffer(self, recorder):
        path = recorder.flush()
        assert path.name == "empty"

    def test_count_returns_buffer_size(self, recorder):
        assert recorder.count() == 0
        recorder.record([{"role": "user", "content": "A"}])
        assert recorder.count() == 1


# ============================================================================
# ToolContext
# ============================================================================

class TestToolContext:
    """Tests for ToolContext — filesystem snapshotting for reward functions."""

    @pytest.fixture
    def ctx(self):
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            yield ToolContext(workdir=workdir)

    def test_init_default_workdir(self):
        ctx = ToolContext()
        assert ctx.workdir == Path.cwd()

    def test_snapshot_before_and_after(self, ctx):
        # Create a file
        (ctx.workdir / "test.txt").write_text("hello")
        ctx.snapshot_before()

        # Modify the file
        (ctx.workdir / "test.txt").write_text("world")
        ctx.snapshot_after()

        assert ctx.file_changed("test.txt") is True

    def test_file_not_changed(self, ctx):
        (ctx.workdir / "stable.txt").write_text("same")
        ctx.snapshot_before()
        ctx.snapshot_after()
        assert ctx.file_changed("stable.txt") is False

    def test_file_created(self, ctx):
        ctx.snapshot_before()
        (ctx.workdir / "new.txt").write_text("new")
        ctx.snapshot_after()

        assert ctx.file_created("new.txt") is True

    def test_file_not_created_if_existed(self, ctx):
        (ctx.workdir / "existing.txt").write_text("old")
        ctx.snapshot_before()
        (ctx.workdir / "existing.txt").write_text("modified")
        ctx.snapshot_after()

        assert ctx.file_created("existing.txt") is False

    def test_record_terminal(self, ctx):
        ctx.record_terminal("Error: disk full")
        ctx.record_terminal("Success: file written")
        assert ctx.grep_output("Error") is True
        assert ctx.grep_output("disk full") is True
        assert ctx.grep_output("missing") is False

    def test_snapshot_excludes_hidden_files(self, ctx):
        (ctx.workdir / ".hidden").write_text("secret")
        (ctx.workdir / "visible.txt").write_text("public")
        ctx.snapshot_before()

        files = ctx._files_before
        assert "visible.txt" in files
        assert ".hidden" not in files

    def test_snapshot_handles_non_file(self, ctx):
        (ctx.workdir / "subdir").mkdir()
        (ctx.workdir / "file.txt").write_text("ok")
        ctx.snapshot_before()
        # Should not error on directories
        assert "file.txt" in ctx._files_before


# ============================================================================
# TrainingEnv
# ============================================================================

class TestTrainingEnv:
    """Tests for TrainingEnv dataclass."""

    def test_default_values(self):
        env = TrainingEnv(name="test-env")
        assert env.name == "test-env"
        assert env.description == ""
        assert env.prompt_template == ""
        assert env.max_turns == 20

    def test_format_prompt(self):
        env = TrainingEnv(
            name="diagnosis",
            prompt_template="Diagnose the issue in {log_file} on {vehicle}",
        )
        result = env.format_prompt(log_file="error.log", vehicle="SUV")
        assert "error.log" in result
        assert "SUV" in result

    def test_format_prompt_empty_template(self):
        env = TrainingEnv(name="test")
        assert env.format_prompt(x="y") == ""

    def test_evaluate_calls_reward_fn(self):
        def my_reward(result, ctx):
            return 42.0

        env = TrainingEnv(name="test", reward_fn=my_reward)
        score = env.evaluate({}, MagicMock())
        assert score == 42.0

    def test_evaluate_catches_exceptions(self):
        def broken_reward(result, ctx):
            raise RuntimeError("boom")

        env = TrainingEnv(name="test", reward_fn=broken_reward)
        score = env.evaluate({}, MagicMock())
        assert score == 0.0

    def test_metadata_stored(self):
        env = TrainingEnv(name="test", metadata={"domain": "automotive"})
        assert env.metadata["domain"] == "automotive"


# ============================================================================
# EnvironmentRegistry
# ============================================================================

class TestEnvironmentRegistry:
    """Tests for EnvironmentRegistry."""

    def test_register_and_get(self):
        registry = EnvironmentRegistry()
        env = TrainingEnv(name="env1")
        registry.register(env)
        assert registry.get("env1") is env

    def test_get_unknown(self):
        registry = EnvironmentRegistry()
        assert registry.get("nonexistent") is None

    def test_list_envs(self):
        registry = EnvironmentRegistry()
        registry.register(TrainingEnv(name="env1"))
        registry.register(TrainingEnv(name="env2"))
        names = registry.list_envs()
        assert "env1" in names
        assert "env2" in names
        assert len(names) == 2

    def test_register_overwrites(self):
        registry = EnvironmentRegistry()
        env1 = TrainingEnv(name="env", description="first")
        env2 = TrainingEnv(name="env", description="second")
        registry.register(env1)
        registry.register(env2)
        assert registry.get("env").description == "second"


# ============================================================================
# RolloutRunner
# ============================================================================

class TestRolloutRunner:
    """Tests for RolloutRunner (rollout execution + reward)."""

    def test_run_records_trajectory_and_reward(self):
        mock_agent = MagicMock()
        mock_agent.run.return_value = {
            "content": "The answer is 42",
            "confidence": 0.95,
            "evidence": [{"step": 1}],
        }

        recorder = TrajectoryRecorder()
        runner = RolloutRunner(agent=mock_agent, recorder=recorder)

        def my_reward(result, ctx):
            return float(result.get("confidence", 0))

        result = runner.run("What is the answer?", reward_fn=my_reward)

        assert result["content"] == "The answer is 42"
        assert result["confidence"] == 0.95
        assert result["reward"] == 0.95
        assert recorder.count() == 1

    def test_run_default_reward_zero(self):
        mock_agent = MagicMock()
        mock_agent.run.return_value = {"content": "ok", "confidence": 0.5}

        recorder = TrajectoryRecorder()
        runner = RolloutRunner(agent=mock_agent, recorder=recorder)

        result = runner.run("Test", reward_fn=None)

        assert result["reward"] == 0.0

    def test_run_batch(self):
        mock_agent = MagicMock()
        mock_agent.run.return_value = {"content": "done", "confidence": 1.0}

        runner = RolloutRunner(agent=mock_agent)
        results = runner.run_batch(["A", "B", "C"])

        assert len(results) == 3
        assert all(r["content"] == "done" for r in results)

    def test_flush_trajectories(self):
        mock_agent = MagicMock()
        mock_agent.run.return_value = {"content": "ok"}

        runner = RolloutRunner(agent=mock_agent)
        runner.run("Test")

        path = runner.flush_trajectories()
        assert path.exists()

    def test_run_includes_duration_ms(self):
        mock_agent = MagicMock()
        mock_agent.run.return_value = {"content": "ok"}

        runner = RolloutRunner(agent=mock_agent)
        result = runner.run("Test")

        assert "duration_ms" in result
        assert result["duration_ms"] >= 0


# ============================================================================
# Built-in Reward Functions
# ============================================================================

class TestRewardFunctions:
    """Tests for built-in reward functions."""

    def test_reward_confidence(self):
        ctx = MagicMock(spec=ToolContext)
        assert reward_confidence({"confidence": 0.8}, ctx) == 0.8
        assert reward_confidence({"confidence": 0.0}, ctx) == 0.0
        assert reward_confidence({}, ctx) == 0.0  # missing key

    def test_reward_confidence_none(self):
        ctx = MagicMock(spec=ToolContext)
        assert reward_confidence({"confidence": None}, ctx) == 0.0

    def test_reward_success_rate(self):
        ctx = MagicMock(spec=ToolContext)
        # Content > 10 chars = success
        assert reward_success_rate({"content": "This is a long enough answer"}, ctx) == 1.0
        # Content too short
        assert reward_success_rate({"content": "short"}, ctx) == 0.0
        # Empty content
        assert reward_success_rate({"content": ""}, ctx) == 0.0

    def test_reward_evidence_quality(self):
        ctx = MagicMock(spec=ToolContext)
        # No evidence
        assert reward_evidence_quality({"evidence": []}, ctx) == 0.0
        # 3 steps → 0.6
        assert reward_evidence_quality({"evidence": [1, 2, 3]}, ctx) == 0.6
        # 5 steps → 1.0
        assert reward_evidence_quality({"evidence": [1, 2, 3, 4, 5]}, ctx) == 1.0
        # 10 steps → capped at 1.0
        assert reward_evidence_quality({"evidence": list(range(10))}, ctx) == 1.0

    def test_reward_file_creation_by_output(self):
        ctx = MagicMock(spec=ToolContext)
        ctx.grep_output = MagicMock(return_value=True)
        assert reward_file_creation({}, ctx) == 1.0

    def test_reward_file_creation_by_filesystem(self):
        ctx = MagicMock(spec=ToolContext)
        ctx.grep_output = MagicMock(return_value=False)
        ctx._files_before = {"a.txt": "hash1"}
        ctx._files_after = {"a.txt": "hash1", "b.txt": "hash2"}
        assert reward_file_creation({}, ctx) == 1.0

    def test_reward_file_creation_no_new_files(self):
        ctx = MagicMock(spec=ToolContext)
        ctx.grep_output = MagicMock(return_value=False)
        ctx._files_before = {"a.txt": "hash1"}
        ctx._files_after = {"a.txt": "hash1"}
        assert reward_file_creation({}, ctx) == 0.0
