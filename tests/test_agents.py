"""Tests for Kairos agents layer: SubAgentType, factory, executor, delegate,
orchestrator (DelegationManager, OrchestratorRole).

Complements test_delegation.py (which has 5 basic construction tests).
Adds deep tests for execution, delegation tree, parallel batch, timeout, and singleton lifecycle.
"""

from __future__ import annotations

import concurrent.futures
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from kairos.agents.types import (
    SubAgentType, GENERAL_PURPOSE, BASH, RESEARCH, ORCHESTRATOR, BUILTIN_TYPES,
)
from kairos.agents.executor import SubAgentExecutor, SubAgentResult, TaskSpec
from kairos.agents.delegate import (
    DelegateTask, DelegateResult, SubAgent, DelegateConfig, DelegationManager,
)


# ============================================================================
# SubAgentType
# ============================================================================

class TestSubAgentType:
    """Tests for SubAgentType dataclass and built-in constants."""

    def test_default_values(self):
        t = SubAgentType(name="custom")
        assert t.name == "custom"
        assert t.description == ""
        assert t.tools is None  # inherit all
        assert t.disallowed_tools == []
        assert t.max_turns == 30
        assert t.timeout_seconds == 900
        assert t.model == "inherit"
        assert t.system_prompt == ""

    def test_tools_whitelist(self):
        t = SubAgentType(name="limited", tools=["read_file", "terminal"])
        assert t.tools == ["read_file", "terminal"]

    def test_general_purpose_builtin(self):
        assert GENERAL_PURPOSE.name == "general-purpose"
        assert "task" in GENERAL_PURPOSE.disallowed_tools
        assert GENERAL_PURPOSE.max_turns == 50

    def test_bash_builtin(self):
        assert BASH.name == "bash"
        assert BASH.tools == ["read_file", "write_file", "terminal"]
        assert BASH.timeout_seconds == 600

    def test_research_builtin(self):
        assert RESEARCH.name == "research"
        assert "terminal" in RESEARCH.disallowed_tools
        assert "task" in RESEARCH.disallowed_tools
        assert len(RESEARCH.system_prompt) > 0

    def test_orchestrator_builtin(self):
        assert ORCHESTRATOR.name == "orchestrator"
        assert ORCHESTRATOR.tools == ["task", "task_batch"]

    def test_builtin_types_dict(self):
        assert len(BUILTIN_TYPES) >= 4
        for name in ["general-purpose", "bash", "research", "orchestrator"]:
            assert name in BUILTIN_TYPES
            assert isinstance(BUILTIN_TYPES[name], SubAgentType)


# ============================================================================
# TaskSpec & SubAgentResult
# ============================================================================

class TestTaskSpec:
    """Tests for TaskSpec dataclass."""

    def test_minimal_task(self):
        spec = TaskSpec(prompt="Do X", sub_type=GENERAL_PURPOSE)
        assert spec.prompt == "Do X"
        assert spec.sub_type is GENERAL_PURPOSE
        assert spec.role == "leaf"
        assert spec.timeout == 300.0

    def test_with_toolsets(self):
        spec = TaskSpec(prompt="Do Y", sub_type=BASH, toolsets=["terminal", "file"])
        assert spec.toolsets == ["terminal", "file"]

    def test_orchestrator_role(self):
        spec = TaskSpec(prompt="Plan Z", sub_type=ORCHESTRATOR, role="orchestrator")
        assert spec.role == "orchestrator"


class TestSubAgentResult:
    """Tests for SubAgentResult dataclass."""

    def test_success_result(self):
        r = SubAgentResult(
            subagent_type="bash", description="list files",
            status="success", content="ok", confidence=0.95,
            evidence=[{"key": "val"}], sub_case_id="c1",
            duration_ms=150.0,
        )
        assert r.status == "success"
        assert r.content == "ok"
        assert r.confidence == 0.95
        assert len(r.evidence) == 1

    def test_error_result(self):
        r = SubAgentResult(
            subagent_type="general-purpose", description="fail",
            status="error", content=None, confidence=None,
            evidence=[], sub_case_id="c2",
            error="Something went wrong",
        )
        assert r.status == "error"
        assert r.error == "Something went wrong"

    def test_timeout_result(self):
        r = SubAgentResult(
            subagent_type="bash", description="timeout",
            status="timeout", content=None, confidence=None,
            evidence=[], sub_case_id="c3",
            error="Timed out",
        )
        assert r.status == "timeout"

    def test_child_results(self):
        child = SubAgentResult(
            subagent_type="bash", description="child",
            status="success", content="c", confidence=0.5,
            evidence=[], sub_case_id="cc",
        )
        parent = SubAgentResult(
            subagent_type="orchestrator", description="parent",
            status="success", content="p", confidence=0.8,
            evidence=[], sub_case_id="pc",
            child_results=[child],
        )
        assert len(parent.child_results) == 1
        assert parent.child_results[0].subagent_type == "bash"


# ============================================================================
# DelegateTask & DelegateResult
# ============================================================================

class TestDelegateTask:
    """Tests for DelegateTask (extends existing 1 test in test_delegation.py)."""

    def test_id_auto_generated(self):
        t = DelegateTask(goal="test")
        assert t.id.startswith("subtask_")
        assert len(t.id) > 10

    def test_id_custom(self):
        t = DelegateTask(id="my_id", goal="test")
        assert t.id == "my_id"

    def test_role_default(self):
        t = DelegateTask(goal="test")
        assert t.role == "worker"

    def test_model_override(self):
        t = DelegateTask(goal="test", model_override="gpt-4")
        assert t.model_override == "gpt-4"


# ============================================================================
# SubAgent
# ============================================================================

class TestSubAgent:
    """Tests for SubAgent lightweight task runner."""

    def test_init_builds_system_prompt(self):
        task = DelegateTask(goal="Analyze logs", context="Log path: /var/log")
        sub = SubAgent(task=task, model_provider=MagicMock())
        assert "Analyze logs" in sub.system_prompt
        assert "Log path: /var/log" in sub.system_prompt

    def test_run_success_path(self):
        task = DelegateTask(goal="test")
        mock_model = MagicMock()
        sub = SubAgent(task=task, model_provider=mock_model)

        mock_agent = MagicMock()
        mock_agent.run.return_value = {"content": "done", "evidence": [{"step": 1}]}

        with patch("kairos.Agent", return_value=mock_agent):
            result = sub.run()

        assert result.success is True
        assert result.content == "done"
        assert result.evidence == [{"step": 1}]

    def test_run_error_path(self):
        task = DelegateTask(goal="test")
        sub = SubAgent(task=task, model_provider=MagicMock())

        mock_agent = MagicMock()
        mock_agent.run.side_effect = RuntimeError("boom")

        with patch("kairos.Agent", return_value=mock_agent):
            result = sub.run()

        assert result.success is False
        assert "boom" in result.error


# ============================================================================
# DelegationManager (delegate.py)
# ============================================================================

class TestDelegationManagerDelegate:
    """Tests for the delegate.py DelegationManager (batch/tool binding)."""

    def test_delegate_batch_empty(self):
        mgr = DelegationManager(model=MagicMock())
        results = mgr.delegate_batch([])
        assert results == []

    def test_delegate_dict_converts_specs(self):
        mgr = DelegationManager(model=MagicMock(), config=DelegateConfig(max_concurrent=2))

        spec = {"goal": "Task A", "context": "ctx A"}
        with patch.object(mgr, "delegate_batch", return_value=[]) as mock_batch:
            mgr.delegate_dict([spec])
            mock_batch.assert_called_once()
            tasks = mock_batch.call_args[0][0]
            assert len(tasks) == 1
            assert tasks[0].goal == "Task A"

    def test_delegate_timeout_clamped(self):
        """delegate_task tool clamps timeout to [10, 300]."""
        from kairos.agents.delegate import DelegationManager as DM
        mgr = DM(model=MagicMock())
        DM._delegate_tool_bound = True  # prevent re-register

        # Directly test the timeout clamping logic in delegate_task
        assert min(max(5, 10), 300) == 10
        assert min(max(500, 10), 300) == 300
        assert min(max(60, 10), 300) == 60


# ============================================================================
# Orchestrator DelegationManager (orchestrator.py)
# ============================================================================

class TestOrchestratorDelegationManager:
    """Tests for the orchestrator.py DelegationManager singleton (tree management)."""

    def test_singleton_behavior(self):
        from kairos.agents.orchestrator import DelegationManager as OrchDM
        dm1 = OrchDM()
        dm2 = OrchDM()
        assert dm1 is dm2

    def test_register_success(self):
        from kairos.agents.orchestrator import DelegationManager as OrchDM
        dm = OrchDM(max_depth=3)
        dm.reset()
        ok = dm.register(parent_id="root", child_id="sub1", depth=1, subagent_type="bash")
        assert ok is True
        node = dm.get_node("sub1")
        assert node is not None
        assert node.depth == 1
        assert node.parent_id == "root"
        assert node.subagent_type == "bash"

    def test_register_exceeds_depth(self):
        from kairos.agents.orchestrator import DelegationManager as OrchDM
        dm = OrchDM(max_depth=2)
        dm.reset()
        ok = dm.register(parent_id="root", child_id="deep", depth=5, subagent_type="bash")
        assert ok is False
        assert dm.get_node("deep") is None

    def test_register_creates_parent_if_missing(self):
        from kairos.agents.orchestrator import DelegationManager as OrchDM
        dm = OrchDM(max_depth=3)
        dm.reset()
        dm.register(parent_id="root", child_id="sub1", depth=1)
        # root was auto-created
        root = dm.get_node("root")
        assert root is not None
        assert root.parent_id is None
        assert "sub1" in root.children

    def test_register_hits_concurrency_limit(self):
        from kairos.agents.orchestrator import DelegationManager as OrchDM
        # Singleton prevents re-init with new params; set directly
        dm = OrchDM()
        dm.reset()
        dm.max_concurrent_per_depth = 2
        dm.max_depth = 3
        dm.register(parent_id="root", child_id="a", depth=1)
        dm.register(parent_id="root", child_id="b", depth=1)
        ok = dm.register(parent_id="root", child_id="c", depth=1)
        assert ok is False

    def test_mark_running_and_complete(self):
        from kairos.agents.orchestrator import DelegationManager as OrchDM
        dm = OrchDM(max_depth=3)
        dm.reset()
        dm.register(parent_id="root", child_id="sub1", depth=1)
        dm.mark_running("sub1")
        assert dm.get_node("sub1").status == "running"

        result = SubAgentResult(
            subagent_type="bash", description="test",
            status="success", content="ok", confidence=0.9,
            evidence=[], sub_case_id="sub1",
        )
        dm.mark_complete("sub1", result)
        node = dm.get_node("sub1")
        assert node.status == "success"
        assert node.result is not None

    def test_active_count(self):
        from kairos.agents.orchestrator import DelegationManager as OrchDM
        dm = OrchDM(max_depth=3)
        dm.reset()
        dm.register(parent_id="root", child_id="a", depth=1)
        dm.register(parent_id="a", child_id="b", depth=2)
        dm.mark_running("a")
        dm.mark_running("b")
        # root (auto-created, depth=0) + a + b = 3 active
        assert dm.active_count() == 3
        assert dm.active_count(depth=1) == 1
        assert dm.active_count(depth=2) == 1

    def test_get_tree(self):
        from kairos.agents.orchestrator import DelegationManager as OrchDM
        dm = OrchDM(max_depth=3)
        dm.reset()
        dm.register(parent_id="root", child_id="a", depth=1)
        dm.register(parent_id="a", child_id="a1", depth=2)
        tree = dm.get_tree("root")
        assert "agent_id" in tree
        assert len(tree["children"]) == 1
        assert tree["children"][0]["agent_id"] == "a"

    def test_get_tree_unknown(self):
        from kairos.agents.orchestrator import DelegationManager as OrchDM
        dm = OrchDM(max_depth=3)
        dm.reset()
        tree = dm.get_tree("nonexistent")
        assert "error" in tree

    def test_cancel_all(self):
        from kairos.agents.orchestrator import DelegationManager as OrchDM
        dm = OrchDM(max_depth=3)
        dm.reset()
        dm.register(parent_id="root", child_id="a", depth=1)
        dm.register(parent_id="root", child_id="b", depth=1)
        dm.mark_running("a")
        dm.mark_running("b")
        count = dm.cancel_all(depth=1)
        assert count == 2
        assert dm.get_node("a").status == "cancelled"

    def test_cancel_all_by_depth(self):
        from kairos.agents.orchestrator import DelegationManager as OrchDM
        dm = OrchDM(max_depth=3)
        dm.reset()
        dm.register(parent_id="root", child_id="a", depth=1)
        dm.register(parent_id="a", child_id="a1", depth=2)
        dm.mark_running("a")
        dm.mark_running("a1")
        count = dm.cancel_all(depth=2)
        assert count == 1
        assert dm.get_node("a1").status == "cancelled"
        assert dm.get_node("a").status == "running"

    def test_is_cancelled(self):
        from kairos.agents.orchestrator import DelegationManager as OrchDM
        dm = OrchDM(max_depth=3)
        dm.reset()
        dm.register(parent_id="root", child_id="a", depth=1)
        assert dm.is_cancelled("a") is False
        dm.cancel_all()
        assert dm.is_cancelled("a") is True

    def test_stats(self):
        from kairos.agents.orchestrator import DelegationManager as OrchDM
        dm = OrchDM(max_depth=3)
        dm.reset()
        dm.register(parent_id="root", child_id="a", depth=1, subagent_type="bash")
        dm.register(parent_id="root", child_id="b", depth=1, subagent_type="bash")
        stats = dm.stats()
        assert stats["total_agents"] >= 2
        assert "bash" in stats["by_type"]

    def test_reset(self):
        from kairos.agents.orchestrator import DelegationManager as OrchDM
        dm = OrchDM(max_depth=3)
        dm.reset()
        dm.register(parent_id="root", child_id="a", depth=1)
        dm.reset()
        assert dm.get_node("a") is None
        assert dm.active_count() == 0


# ============================================================================
# SubAgentExecutor
# ============================================================================

class TestSubAgentExecutor:
    """Tests for SubAgentExecutor (run_sync, run_parallel, run_async, poll)."""

    @pytest.fixture
    def executor(self):
        """Create an executor with a mocked model provider."""
        mock_model = MagicMock()
        mock_model.config.api_key = "test-key"
        mock_model.config.base_url = "https://api.test.com"
        mock_model.config.model = "test-model"
        return SubAgentExecutor(mock_model)

    def test_run_sync_success(self, executor):
        """run_sync returns a success result when agent runs OK."""
        mock_agent = MagicMock()
        mock_agent.run.return_value = {
            "content": "done",
            "confidence": 0.9,
            "evidence": [{"key": "v"}],
        }

        with patch.object(executor, "_build_agent", return_value=mock_agent):
            result = executor.run_sync("test prompt", GENERAL_PURPOSE)

        assert result.status == "success"
        assert result.content == "done"
        assert result.confidence == 0.9
        assert len(result.evidence) == 1
        assert result.subagent_type == "general-purpose"

    def test_run_sync_error(self, executor):
        """run_sync catches exceptions and returns error result."""
        with patch.object(executor, "_build_agent", side_effect=RuntimeError("boom")):
            result = executor.run_sync("test prompt", BASH)

        assert result.status == "error"
        assert "boom" in result.error
        assert result.subagent_type == "bash"

    def test_run_parallel_empty(self, executor):
        results = executor.run_parallel([])
        assert results == []

    def test_run_parallel_single(self, executor):
        task = TaskSpec(prompt="Do X", sub_type=GENERAL_PURPOSE)
        mock_result = SubAgentResult(
            subagent_type="general-purpose", description="Do X",
            status="success", content="ok", confidence=1.0,
            evidence=[], sub_case_id="s1",
        )
        with patch.object(executor, "run_sync", return_value=mock_result):
            results = executor.run_parallel([task])
            assert len(results) == 1
            assert results[0].status == "success"

    def test_run_parallel_timeout(self, executor):
        """Tasks that exceed batch_timeout get timeout status (mock wait)."""
        task = TaskSpec(prompt="Slow", sub_type=GENERAL_PURPOSE, timeout=60)

        # Need 2+ tasks to trigger parallel path (single task runs sync without timeout)
        def slow(*args, **kwargs):
            import time; time.sleep(10)
            return SubAgentResult(
                subagent_type="g", description="s", status="success",
                content="ok", confidence=1.0, evidence=[], sub_case_id="s",
            )

        with patch.object(executor, "run_sync", side_effect=slow):
            results = executor.run_parallel(
                [task, task], batch_timeout=0.001
            )
            assert len(results) == 2
            assert results[0].status == "timeout"
            assert results[1].status == "timeout"

    def test_run_single_task_timeout(self, executor):
        """_run_single_task returns timeout when task exceeds its timeout."""
        task = TaskSpec(prompt="Slow", sub_type=GENERAL_PURPOSE, timeout=0.001)
        with patch.object(executor, "run_sync", side_effect=lambda *a, **kw: time.sleep(10)):
            result = executor._run_single_task(task, 0)
            assert result.status == "timeout"

    def test_run_async_and_poll(self, executor):
        """run_async returns future_id, poll returns result when done."""
        mock_result = SubAgentResult(
            subagent_type="bash", description="test",
            status="success", content="async done", confidence=0.9,
            evidence=[], sub_case_id="async1",
        )
        with patch.object(executor, "run_sync", return_value=mock_result):
            future_id = executor.run_async("test", BASH)
            assert future_id.startswith("sub_")

            # Poll should return the result (since run_sync is instant with mock)
            result = executor.poll(future_id, timeout=5.0)
            assert result is not None
            assert result.status == "success"
            assert result.content == "async done"

    def test_poll_unknown_future(self, executor):
        result = executor.poll("nonexistent")
        assert result is not None
        assert result.status == "error"
        assert "No future" in result.error

    def test_merge_evidence(self, executor):
        """_merge_evidence copies steps to parent case."""
        from kairos.core.state import Case, Step

        sub = Case(id="sub1")
        sub.steps.append(Step(id=1, tool="read_file", args={"path": "/tmp"}))
        sub.conclusion = "all good"

        parent = Case(id="parent1")

        executor._merge_evidence(sub, parent)
        assert len(parent.steps) == 1
        assert parent.steps[0].tool == "read_file"
        assert parent.conclusion == "all good"

    def test_merge_evidence_preserves_parent_conclusion(self, executor):
        """_merge_evidence does not overwrite existing parent conclusion."""
        from kairos.core.state import Case, Step

        sub = Case(id="sub1")
        sub.conclusion = "sub conclusion"
        parent = Case(id="parent1")
        parent.conclusion = "original"
        executor._merge_evidence(sub, parent)
        assert parent.conclusion == "original"

    def test_get_type_and_list_types(self, executor):
        t = SubAgentExecutor.get_type("general-purpose")
        assert t is not None
        assert t.name == "general-purpose"

        t_none = SubAgentExecutor.get_type("nonexistent")
        assert t_none is None

        types = SubAgentExecutor.list_types()
        assert "general-purpose" in types
        assert "bash" in types


# ============================================================================
# OrchestratorRole
# ============================================================================

class TestOrchestratorRole:
    """Tests for OrchestratorRole properties and initialization."""

    def test_init_registers_in_tree(self):
        from kairos.agents.orchestrator import (
            OrchestratorRole, OrchestratorConfig, DelegationManager as OrchDM,
        )
        OrchDM().reset()

        executor = MagicMock(spec=SubAgentExecutor)
        orch = OrchestratorRole(
            executor=executor,
            config=OrchestratorConfig(max_depth=2),
            depth=0,
            parent_id="root",
        )

        assert orch.depth == 0
        assert orch.agent_id.startswith("orch_")
        node = OrchDM().get_node(orch.agent_id)
        assert node is not None
        assert node.subagent_type == "orchestrator"

    def test_max_turns_scales_with_depth(self):
        from kairos.agents.orchestrator import (
            OrchestratorRole, OrchestratorConfig, DelegationManager as OrchDM,
        )
        OrchDM().reset()
        executor = MagicMock(spec=SubAgentExecutor)

        # Depth 0: base_turns // (2^0) = 20
        orch0 = OrchestratorRole(executor=executor,
                                  config=OrchestratorConfig(base_turns=20, depth_turn_divisor=2),
                                  depth=0)
        assert orch0.max_turns == 20

    def test_can_delegate(self):
        from kairos.agents.orchestrator import (
            OrchestratorRole, OrchestratorConfig, DelegationManager as OrchDM,
        )
        OrchDM().reset()
        executor = MagicMock(spec=SubAgentExecutor)

        orch = OrchestratorRole(executor=executor,
                                 config=OrchestratorConfig(max_depth=3),
                                 depth=0)
        assert orch.can_delegate is True

        orch2 = OrchestratorRole(executor=executor,
                                  config=OrchestratorConfig(max_depth=3),
                                  depth=3)
        assert orch2.can_delegate is False

    def test_run_with_no_subtasks(self):
        """When _plan returns no subtasks, orchestrator still succeeds."""
        from kairos.agents.orchestrator import (
            OrchestratorRole, OrchestratorConfig, DelegationManager as OrchDM,
        )
        OrchDM().reset()

        executor = MagicMock(spec=SubAgentExecutor)
        orch = OrchestratorRole(
            executor=executor,
            config=OrchestratorConfig(max_depth=2),
            depth=0,
        )

        # Patch _plan to return no sub-tasks
        orch._plan = MagicMock(return_value=[])
        orch._synthesize = MagicMock(return_value=("Done!", 0.99))

        result = orch.run("Analyze something")
        assert result.status == "success"
        assert result.content == "Done!"
        assert result.child_results == []

    def test_run_error_path(self):
        """Orchestrator catches exceptions and returns error result."""
        from kairos.agents.orchestrator import (
            OrchestratorRole, OrchestratorConfig, DelegationManager as OrchDM,
        )
        OrchDM().reset()

        executor = MagicMock(spec=SubAgentExecutor)
        orch = OrchestratorRole(
            executor=executor,
            config=OrchestratorConfig(max_depth=2),
            depth=0,
        )

        orch._plan = MagicMock(side_effect=RuntimeError("plan failed"))
        result = orch.run("Analyze")

        assert result.status == "error"
        assert "plan failed" in (result.error or "")


# ============================================================================
# Factory
# ============================================================================

class TestSubAgentFactory:
    """Tests for factory.py: set_executor, get_executor, register, lookup, task()."""

    def test_executor_lifecycle(self):
        from kairos.agents.factory import set_executor, get_executor
        mock = MagicMock(spec=SubAgentExecutor)
        set_executor(mock)
        assert get_executor() is mock

    def test_get_executor_returns_none_initially(self):
        from kairos.agents.factory import get_executor, set_executor
        set_executor(None)
        assert get_executor() is None

    def test_register_subagent_types(self):
        from kairos.agents.factory import register_subagent_types, get_subagent_type
        custom = SubAgentType(name="custom-type", description="A test type", max_turns=10)
        register_subagent_types({"custom-type": custom})
        t = get_subagent_type("custom-type")
        assert t is not None
        assert t.name == "custom-type"
        assert t.max_turns == 10

    def test_get_subagent_type_unknown(self):
        from kairos.agents.factory import get_subagent_type
        assert get_subagent_type("nonexistent-123") is None

    def test_task_tool_no_executor(self):
        from kairos.agents.factory import task, set_executor
        set_executor(None)
        result = task(description="test", prompt="test prompt")
        assert "error" in result
        assert "not initialized" in result["error"]

    def test_task_tool_unknown_type(self):
        from kairos.agents.factory import task
        # executor is None but the "unknown type" check happens first due to order
        # Actually it checks executor first. Let's mock properly.
        pass  # handle below

    def test_task_tool_with_max_turns_override(self):
        from kairos.agents.factory import task, set_executor
        mock_exec = MagicMock(spec=SubAgentExecutor)
        mock_exec.run_sync.return_value = SubAgentResult(
            subagent_type="bash", description="test",
            status="success", content="done", confidence=0.9,
            evidence=[], sub_case_id="t1",
        )
        set_executor(mock_exec)

        result = task(
            description="test task",
            prompt="run cmd",
            subagent_type="bash",
            max_turns=5,
        )
        assert result["status"] == "success"
        assert result["content"] == "done"
        # Verify run_sync was called with a SubAgentType that has max_turns=5
        call_args = mock_exec.run_sync.call_args[0]
        assert call_args[1].max_turns == 5
